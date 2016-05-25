import numpy

import numpy
from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping, Transformer, SourcewiseTransformer)

from six.moves import cPickle

from machine_translation.stream import (_ensure_special_tokens, _length, PaddingWithEOS, _oov_to_unk, _too_long,
                                        ShuffleBatchTransformer)


def imt_f1(hyp, ref):
    """
    compute Ueffing and Ney F1 for IMT

    Note that this function is agnostic about its inputs, as long as they
    are sequences. Thus the metric can be computed for sequences of characters,
    words, phrases, etc...

    :returns f1_score, precision, recall

    """

    # if both are empty, this is a perfect match
    if len(hyp) == 0 and len(ref) == 0:
        return 1., 1., 1.

    match_len = float(0)
    hyp_len = float(len(hyp))
    ref_len = float(len(ref))
    for h_sym, r_sym in zip(hyp, ref):
        if h_sym == r_sym:
            match_len += 1.
        else:
            break

    if match_len == 0:
        return 0., 0., 0.

    # ratio of characters in the prediction which are correct (low if prefix is too long)
    precision = match_len / hyp_len

    # ratio of coverage of the reference (low if prefix is too short)
    recall = match_len / ref_len
    return 2 * ((precision * recall) / (precision + recall)), precision, recall


def dcg(scores):
    num_scores = len(scores)
    assert num_scores > 0, 'you must pass a 1d iterable containing at least one score'

    scaled_scores = []
    for s, i in zip(scores, range(1, num_scores + 1)):
        scaled_score = ((2**s) - 1) / numpy.log2(i + 1)
        scaled_scores.append(scaled_score)
    return numpy.sum(scaled_scores)


def ndcg(scores):
    ideal_dcg = dcg(sorted(scores, reverse=True))
    normed_cdg = dcg(scores) / ideal_dcg
    return normed_cdg


def map_pair_to_imt_triples(source, reference, bos_token=None, eos_token=None):
    """
    Map a (source, reference) pair into (len(actual_reference) + 2) new examples

    Assumes that users always want the empty prefix at the beginning of the generated examples
    (i.e. ask the system for the full hypothesis) and the empty suffix (i.e. ask the system for nothing)
    at the end of the generated examples

    By passing None for bos_token or eos_token, user indicates that these tokens have already
    been prepended or appended to the reference

    Note: may want to refactor this into a function which takes just the target and returns (prefix, suffix) pairs
    for more flexibility

    """

    start_index = 0
    if bos_token:
        start_index = 1

    end_index = len(reference) + 1
    if eos_token:
        end_index -= 1

    prefixes, suffixes = zip(*[(reference[:i], reference[i:]) for i in range(start_index, end_index)])
    sources = [source for _ in range(end_index - start_index)]

    assert len(sources) == len(prefixes) == len(suffixes), 'All components must have the same length'

    return zip(sources, prefixes, suffixes)


# Module for functionality associated with streaming data
class PrefixSuffixStreamTransformer:
    """
    Takes a stream of (source, target) and adds the sources ('suffixes', 'prefixes'),



    Parameters
    ----------


    Notes
    -----
    At call time, we expect a stream providing (sources, references) -- i.e. something like a TextFile object

    In the future, we may want to randomly provide _some_ of the prefix/suffix pairs for a target sequence, but not
    all of them, this would require initializing the transformer with some params.

    Note that we also need to duplicate the source and target the required number of times (this depends upon
    the length of the target sequence). Currently this is accomplished in a separate transformer
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data, **kwargs):
        source = data[0]
        reference = data[1]

        # TODO: we need to pass through the information about BOS and EOS tokens
        # TODO: there is wasted computation here, since we will need to flatten the sources back out again later
        sources, target_prefixes, target_suffixes = zip(*map_pair_to_imt_triples(source, reference,
                                                                                 bos_token=True,
                                                                                 eos_token=True,
                                                                                 **kwargs))

        # Note: the cast here is important, otherwise these will become float64s which will break everything
        target_prefixes = [numpy.array(pre).astype('int64') for pre in target_prefixes]
        target_suffixes = [numpy.array(suf).astype('int64') for suf in target_suffixes]

        return (target_prefixes, target_suffixes)


class CopySourceAndTargetToMatchPrefixes(Transformer):
    """Duplicate the source and target to match the number of prefixes and suffixes

    We need this transformer because Fuel does not directly support transformers which _BOTH_ add sources _and_
    modify existing sources

    Parameters
    ----------
    data_stream : :class:`AbstractDataStream` instance
        The data stream to wrap

    Notes
    -----

    It isn't very nice how all of the source names are hard-coded

    """

    @property
    def sources(self):
        # data_stream should be an attribute of the parent class
        return self.data_stream.sources

    def __call__(self, data, **kwargs):
        batch_obj = {k:v for k,v in zip(self.data_stream.sources, data)}
        num_prefixes = len(batch_obj['target_prefix'])

        batch_obj['source'] = tuple([batch_obj['source'] for _ in range(num_prefixes)])
        batch_obj['target'] = tuple([batch_obj['target'] for _ in range(num_prefixes)])

        batch_with_expanded_source_and_target = [batch_obj[k] for k in self.data_stream.sources]

        return tuple(batch_with_expanded_source_and_target)


def get_tr_stream_with_prefixes(src_vocab, trg_vocab, src_data, trg_data, src_vocab_size=30000,
                                trg_vocab_size=30000, unk_id=1, seq_len=50,
                                batch_size=80, sort_k_batches=12, **kwargs):
    """Prepares the IMT training data stream."""

    # Load dictionaries and ensure special tokens exist
    src_vocab = _ensure_special_tokens(
        src_vocab if isinstance(src_vocab, dict)
        else cPickle.load(open(src_vocab)),
        bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
    trg_vocab = _ensure_special_tokens(
        trg_vocab if isinstance(trg_vocab, dict) else
        cPickle.load(open(trg_vocab)),
        bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)

    # Get text files from both source and target
    src_dataset = TextFile([src_data], src_vocab,
                           bos_token='<S>',
                           eos_token='</S>',
                           unk_token='<UNK>')
    trg_dataset = TextFile([trg_data], trg_vocab,
                           bos_token='<S>',
                           eos_token='</S>',
                           unk_token='<UNK>')

    # Merge them to get a source, target pair
    stream = Merge([src_dataset.get_example_stream(),
                    trg_dataset.get_example_stream()],
                   ('source', 'target'))

    # Filter sequences that are too long
    stream = Filter(stream,
                    predicate=_too_long(seq_len=seq_len))

    # Replace out of vocabulary tokens with unk token
    # TODO: doesn't the TextFile stream do this anyway?
    stream = Mapping(stream,
                     _oov_to_unk(src_vocab_size=src_vocab_size,
                                 trg_vocab_size=trg_vocab_size,
                                 unk_id=unk_id))

    #
    stream = Mapping(stream, PrefixSuffixStreamTransformer(),
                     add_sources=('target_prefix', 'target_suffix'))

    stream = Mapping(stream, CopySourceAndTargetToMatchPrefixes(stream))

    # changing stream.produces_examples is a little hack which lets us use Unpack to flatten
    stream.produces_examples = False
    # flatten the stream back out into (source, target, target_prefix, target_suffix)
    stream = Unpack(stream)

    # Now make a very big batch that we can shuffle
    # Build a batched version of stream to read k batches ahead
    shuffle_batch_size = kwargs['shuffle_batch_size']
    stream = Batch(stream,
                   iteration_scheme=ConstantScheme(shuffle_batch_size)
                   )

    stream = ShuffleBatchTransformer(stream)

    # unpack it again
    stream = Unpack(stream)

    # Build a batched version of stream to read k batches ahead
    stream = Batch(stream,
                   iteration_scheme=ConstantScheme(batch_size * sort_k_batches)
                   )

    # Sort all samples in the read-ahead batch
    stream = Mapping(stream, SortMapping(_length))

    # Convert it into a stream again
    stream = Unpack(stream)

    # Construct batches from the stream with specified batch size
    stream = Batch(
        stream, iteration_scheme=ConstantScheme(batch_size))

    # Pad sequences that are short
    # TODO: is it correct to blindly pad the target_prefix and the target_suffix?
    masked_stream = PaddingWithEOS(
        stream, [src_vocab_size - 1, trg_vocab_size - 1, trg_vocab_size - 1, trg_vocab_size - 1],
        mask_sources=('source', 'target', 'target_prefix', 'target_suffix'))

    return masked_stream, src_vocab, trg_vocab


# # Remember that the BleuValidator does hackish stuff to get target set information from the main_loop data_stream
# # using all kwargs here makes it more clear that this function is always called with get_dev_stream(**config_dict)
def get_dev_stream_with_prefixes(val_set=None, val_set_grndtruth=None, src_vocab=None, src_vocab_size=30000,
                                 trg_vocab=None, trg_vocab_size=30000, unk_id=1, return_vocab=False, **kwargs):
    """Setup development set stream if necessary."""

    dev_stream = None
    if val_set is not None and val_set_grndtruth is not None:
        src_vocab = _ensure_special_tokens(
            src_vocab if isinstance(src_vocab, dict) else
            cPickle.load(open(src_vocab)),
            bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)

        trg_vocab = _ensure_special_tokens(
            trg_vocab if isinstance(trg_vocab, dict) else
            cPickle.load(open(trg_vocab)),
            bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)

        dev_source_dataset = TextFile([val_set], src_vocab,
                                      bos_token='<S>',
                                      eos_token='</S>',
                                      unk_token='<UNK>')
        dev_target_dataset = TextFile([val_set_grndtruth], trg_vocab,
                                      bos_token='<S>',
                                      eos_token='</S>',
                                      unk_token='<UNK>')

        dev_stream = Merge([dev_source_dataset.get_example_stream(),
                            dev_target_dataset.get_example_stream()],
                           ('source', 'target'))

        # now add prefix and suffixes to this stream
        dev_stream = Mapping(dev_stream, PrefixSuffixStreamTransformer(),
                         add_sources=('target_prefix', 'target_suffix'))

        dev_stream = Mapping(dev_stream, CopySourceAndTargetToMatchPrefixes(dev_stream))

        # changing stream.produces_examples is a little hack which lets us use Unpack to flatten
        dev_stream.produces_examples = False
        # flatten the stream back out into (source, target, target_prefix, target_suffix)
        dev_stream = Unpack(dev_stream)

    if return_vocab:
        return dev_stream, src_vocab, trg_vocab
    else:
        return dev_stream



