import numpy
import logging
import copy

from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping, Transformer, SourcewiseTransformer)

from six.moves import cPickle

from machine_translation.stream import (_ensure_special_tokens, _length, PaddingWithEOS, _oov_to_unk, _too_long,
                                        ShuffleBatchTransformer)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _length(sentence_pair):
    """Assumes suffix is the fourth element in the tuple."""
    return len(sentence_pair[3])

def n_constraints_from_sequence(seq, num_constraints):
    """
    Select N non-overlapping constraints from sequence, use heuristics to determine the max constraint length

    :param ref_seq:
    :param ref_seq:
    :return:
    """

    seq_len = len(seq)
    # constraints ideally should not cover more than half of the reference (the logic below doens't enforce this, just 'recommends'
    max_constraint_len = int(numpy.ceil(seq_len / 2. / num_constraints))
    possible_constraint_lens = range(1, max_constraint_len + 1)
    constraint_lens = numpy.random.choice(possible_constraint_lens, num_constraints)

    # max constraint len is 1, so this sequence is short, make sure constraints won't try to go beyond end of seq
    if len(possible_constraint_lens) == 1:
        constraint_lens = numpy.array(constraint_lens[:seq_len - 2])

    # go through the constraints, select a random start point from the remaining options, if you finish without getting all, some constraints are empty
    constraints = []
    constraint_idxs = []
    start_idx = 1
    for c_idx, c_len in enumerate(constraint_lens):
        start_window = range(start_idx, max(start_idx + 1, seq_len - constraint_lens[c_idx:].sum()))
        c_start_idx = numpy.random.choice(start_window)
        c_end_idx = c_start_idx + c_len
        constraints.append(seq[c_start_idx:c_end_idx])
        constraint_idxs.append(range(c_start_idx, c_end_idx))
        start_idx = c_end_idx

    return constraints, constraint_idxs


# WORKING: transformer which takes source + target, and adds `constraints` `constraints_mask`, `model_choice_sequence`
# WORKING: note that this transformer expects the _FULL_ target as reference, not just the suffix
# WORKING: note that model choice sequence needs to be padded with 0, not `EOS`, because this is a sequence of ints
# WORKING: to start, we can just hack the prefix transformer and add the `model_choice_sequence` as well
class ConstraintModelStreamTransformer:
    """
    Takes a stream of (source, target) and adds the sources ('suffixes', 'prefixes', model_choice_sequence),

    Note: TODO: currently the source names are hacked for convenience, later we will generalize this
    #Takes a stream of (source, target) and adds the sources ('constraints', 'model_choice_sequence'),


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
        # self.random_seed = kwargs.get('random_seed', 42)
        # self.random_state = numpy.random.RandomState(self.random_seed)
        # whether we should always generate the same samples
        # TODO: there is an error in the logic here, see below
        # self.static_samples = kwargs.get('static_samples', False)
        # self.do_not_expand = kwargs.get('nmt_baseline_training', False)

        self.max_num_constraints = kwargs.get('max_num_constraints', 3)


    # TODO: reset `suffix` to the full target sequence -- optionally with special tokens inserted
    def __call__(self, data, **kwargs):
        source = data[0]
        reference = data[1]

        num_constraints = numpy.random.choice(range(1, self.max_num_constraints + 1))
        constraints, constraint_idxs = n_constraints_from_sequence(seq=reference, num_constraints=num_constraints)

        # TODO: this is a hook to allow fallback to baseline training
        # TODO: should be removed from constraint model
        # if self.do_not_expand:
        #     sample_idxs=[0]

        # TODO: add the 'within constraint' index, add the 'which constraint' feature -- i.e. 'C1-2-tok'
        # TODO: each of these features has its own embedding, concat embeddings to get the final representation
        # 0 is the generator model, 1 is the pointer model
        flat_constraint_idxs = [idx for cons in constraint_idxs for idx in cons]
        flat_constraints = [c_tok for cons in constraints for c_tok in cons]
        model_choice_sequence = numpy.zeros(len(reference), dtype='float32')
        model_choice_sequence[flat_constraint_idxs] = 1.

        mapped_reference = numpy.array(reference, dtype='int64')

        # here we are mapping the indexes that are constraints to their indexes from the attention model
        mapped_reference[flat_constraint_idxs] = numpy.array(len(flat_constraint_idxs))

        # make everything into size = 1 lists
        flat_constraints = [flat_constraints]
        mapped_reference = [mapped_reference]
        model_choice_sequence = [model_choice_sequence]

        return (flat_constraints, mapped_reference, model_choice_sequence)


# adds a transformer to concat source + <B-CONSTRAINT_i> target <E-CONSTRAINT_i>
# TODO: tokens for B-CONSTRAINT_i and E-CONSTRAINT_i up to N constraints
# TODO: PLACEHOLDER-i for target tokens corresponding to *-CONSTRAINT-i tokens
# TODO: rename to "Concat source and prefix transformer"
class SourceAndPrefixTransformer:
    """
    Takes a stream of <FILL IN ALL SOURCES> and adds the source: `source_with_constraints`


    Parameters
    ----------


    Notes
    -----
    At call time, the provided stream must have the sources in the following order: (<FILLIN>)
    """

    def __init__(self, begin_constraint_idx, end_constraint_idx, gap_idx, vocab_dict=None, **kwargs):
        # TODO: provide the source indices in the kwargs (which idxs are the source, prefix, and suffix)
        # TODO: provide the GAP and CONSTRAINT token format in the kwargs -- let user specify what kind(s) of gaps and constraints are available to the model
        # the final sources are: ('source', 'source_mask', 'target', 'target_mask', 'target_prefix', 'target_prefix_mask', 'target_suffix', 'target_suffix_mask')

        # TODO: this transformer must know the indexes of the constraint and GAP tokens
        # TODO: this will have to change when we support multiple constraints
        self.num_constraints = 1
        self.begin_constraint_idx = begin_constraint_idx
        self.end_constraint_idx = end_constraint_idx
        self.gap_idx = gap_idx

    def __call__(self, data, **kwargs):
        """
        Assumes the sources in `data` are: ('source', 'target', 'target_prefix', 'target_suffix')
        This is intended to be used with Fuel's `Mapping` transformer
        """
        data_image = list(data)
        source = data_image[0]
        target_prefix = data_image[2]
        target_suffix = data_image[3]

        # WORKING: the implicit assumption is that we're going to totally discard the target_prefix in the model -- all constraint information will be available via the attention only
        # WORKING: build the source + constraint representation
        # TODO: this transformer must know the indexes of the constraint and GAP tokens
        #source = np.array(source + [begin_constraint_prefix + '0'] + target_prefix + [self.end_constraint_prefix + '0']
        source = numpy.array(list(source) + [self.begin_constraint_idx] + list(target_prefix) + [self.end_constraint_idx])
        target_suffix = numpy.array([self.gap_idx] + list(target_suffix))
        data_image[0] = source
        data_image[3] = target_suffix

        return tuple(data_image)


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
        self.sample_ratio = kwargs.get('sample_ratio', 1.)
        # TODO: add the ability to always generate the same samples by resetting the seed
        assert self.sample_ratio > 0. and self.sample_ratio <= 1., '0. < sample_ratio <= 1.'

        # only sample suffixes above a certain length
        self.min_suffix_source_ratio = kwargs.get('min_suffix_source_ratio', None)
        if self.min_suffix_source_ratio is not None:
            assert self.min_suffix_source_ratio > 0. and self.min_suffix_source_ratio <= 1., '0. < min_suffix_source_ratio <= 1.'

        self.random_seed = kwargs.get('random_seed', 42)
        self.random_state = numpy.random.RandomState(self.random_seed)

        # whether we should always generate the same samples
        # TODO: there is an error in the logic here, see below
        self.static_samples = kwargs.get('static_samples', False)
        self.do_not_expand = kwargs.get('nmt_baseline_training', False)

    def __call__(self, data, **kwargs):
        source = data[0]
        reference = data[1]

        # Note: there is wasted computation here, since we will need to flatten the sources back out again later
        sources, target_prefixes, target_suffixes = zip(*map_pair_to_imt_triples(source, reference,
                                                                                 bos_token=True,
                                                                                 eos_token=True,
                                                                                 **kwargs))

        # TODO: HACK here -- pairs should be pre-filtered so that the ratio cannot be crazy skewed
        if self.min_suffix_source_ratio is not None and (float(len(reference)) / float(len(source))) >= 0.8:
            source_len = float(len(sources[0]))
            good_idxs = [idx for idx, seq in enumerate(target_suffixes)
                         if (float(len(seq)) / source_len) >= self.min_suffix_source_ratio]

            sources = [sources[idx] for idx in good_idxs]
            target_prefixes = [target_prefixes[idx] for idx in good_idxs]
            target_suffixes = [target_suffixes[idx] for idx in good_idxs]
            if len(good_idxs) == 0:
                import ipdb; ipdb.set_trace()

        if self.sample_ratio < 1.:
            num_samples = int(numpy.ceil(self.sample_ratio * len(target_prefixes)))
            sample_idxs = self.random_state.choice(range(len(target_prefixes)), num_samples)
        else:
            sample_idxs = range(len(target_prefixes))

        # Note: this could be done before expansion to save computation
        if self.do_not_expand:
            sample_idxs=[0]

        logging.info('Generating {} samples, overall len is {}'.format(len(sample_idxs), len(target_prefixes)))

        # Note: the cast here is important, otherwise these will become float64s which will break everything
        target_prefixes = list(numpy.array([numpy.array(pre).astype('int64') for pre in target_prefixes])[sample_idxs])
        target_suffixes = list(numpy.array([numpy.array(suf).astype('int64') for suf in target_suffixes])[sample_idxs])

        # TODO: this won't work, we actually want to reset it after a complete epoch
        # TODO: check calls to reset()? possibly overload that
        # if user wants static samples, reset the random state so that the samples will be the same next time around
        # if self.static_samples == True:
        #     self.random_state = numpy.random.RandomState(self.random_seed)

        return (target_prefixes, target_suffixes)


class CallPredictionFunctionOnStream:
    """
    Init with a function, and let the user specify the indices (and order of indices) which will be used to call the
    function

    Parameters
    ----------
    function: function: the function that will be called
    arg_indices: list: an ordering specifying the indices of the datastream that should be used to call this function

    Notes
    -----

    """

    def __init__(self, function, arg_indices):
        self.function = function
        self.arg_indices = arg_indices

    def __call__(self, data, **kwargs):

        # Note: remember that the convention is for a theano function to return a list, so `output` is always a list
        output = self.function(*[data[idx] for idx in self.arg_indices])

        # HACK: this is specific to the confidence model usecase
        predictions = output[0].argmax(axis=-1).T
        tags = (predictions == data[6]).astype('float32')

        # Note: features which could be added
        # add softmax score of the chosen tag to the output
        # add features for source len, prefix len, position in suffix (position in suffix only makes sense if we're training on predictions)

        # (time, batch, vocab)
        exp_output = numpy.exp(output[0])
        softmax_probs = exp_output / numpy.repeat(numpy.sum(exp_output, axis=-1)[:,:,None], exp_output.shape[-1], axis=-1)
        softmax_feature = numpy.max(softmax_probs, axis=-1)[:, :, None]
        all_features = numpy.concatenate([output[1], softmax_feature], axis=-1)

        # return (predictions, output[1], tags)
        return (predictions, all_features, tags)


# Module for functionality associated with streaming data
class IMTSampleStreamTransformer:
    """
    Stateful transformer which takes a stream of (source, target) and adds the sources ('samples', 'scores')

    Samples are generated by calling the sample func with the source as argument

    Scores are generated by comparing each generated sample to the reference

    Parameters
    ----------
    sample_func: function(num_samples=1) which takes source seq and outputs <num_samples> samples
    score_func: function

    At call time, we expect a stream providing (sources, references) -- i.e. something like a TextFile object


    """

    def __init__(self, sample_func, score_func, num_samples=1, max_value=0.99, min_value=0.1, **kwargs):
        self.sample_func = sample_func
        self.score_func = score_func
        self.num_samples = num_samples
        self.max_value = max_value
        self.min_value = min_value
        # kwargs will get passed to self.score_func when it gets called
        self.kwargs = kwargs

    def __call__(self, data, **kwargs):
        source = data[0]
        reference = data[1]
        prefix = data[2]
        suffix = data[3] # in IMT, the suffix is the reference

        # each sample may be of different length
        samples, seq_probs = self.sample_func(numpy.array(source), numpy.array(prefix), self.num_samples)
        assert len(samples) == len(seq_probs), 'we must have one probability score per sample'

        # TODO: here we need to check for (1) duplicate samples, and (2) add the reference to the sample set
        # TODO: finding non-duplicate samples could loop infinitely, so we should add a max_tries param

        # Note: we currently have to pass the source because of the interface to mteval_v13
        scores = self._compute_scores(source, suffix, samples, **self.kwargs)
        print('raw scores for this sample: {}'.format(scores))

        filtered_scores = []
        for i,s in enumerate(scores):
            if s < self.min_value:
                filtered_scores.append(self.min_value)
            elif s > self.max_value:
                filtered_scores.append(self.max_value)
            else:
                filtered_scores.append(s)

        filtered_scores = numpy.array(filtered_scores, dtype='float32')
        print('filtered scores for this sample: {}'.format(scores))

        # Note: we might get a nan here
        test_probs = (seq_probs**0.005) / (seq_probs**0.005).sum()
        test_expectations = (test_probs * filtered_scores).sum()
        print('expected score for this sample: {}'.format(test_expectations))

        try:
            assert numpy.any(numpy.isnan(test_expectations)) == False, 'there must _not_ be any nans in the expected scores'
        except AssertionError:
            import ipdb;ipdb.set_trace()

        print('source: {}'.format(source))
        print('prefix: {}'.format(prefix))
        print('suffix: {}'.format(suffix))
        print('samples: {}'.format(samples))
        print('scores: {}'.format(filtered_scores))

        return (samples, seq_probs, filtered_scores)

    # Note that many sentence-level metrics like BLEU can be computed directly over the indexes (not the strings),
    # Note that some sentence-level metrics like METEOR require the string representation
    # if the scoring function needs to map from ints to strings, provide 'src_vocab' and 'trg_vocab' via the kwargs
    # So we don't need to map back to a string representation
    def _compute_scores(self, source, reference, samples, **kwargs):
        """Call the scoring function to compare each sample to the reference"""

        return self.score_func(source, reference, samples, **kwargs)


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

        num_samples = len(batch_obj['target_prefix'])

        batch_obj['source'] = tuple([batch_obj['source'] for _ in range(num_samples)])
        batch_obj['target'] = tuple([batch_obj['target'] for _ in range(num_samples)])

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

    # Note: Whether we should use BOS and EOS tokens actually depends upon how the system was pre-trained,
    # Note: but systems used for initialization should _always_ have BOS tokens
    # Get text files from both source and target
    src_dataset = TextFile([src_data], src_vocab,
                           bos_token=u'<S>',
                           eos_token=u'</S>',
                           unk_token=u'<UNK>',
                           encoding='utf8')
    trg_dataset = TextFile([trg_data], trg_vocab,
                           bos_token=u'<S>',
                           eos_token=u'</S>',
                           unk_token=u'<UNK>',
                           encoding='utf8')

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

    # Note: the semantics of the 'target_prefix' variable are completely wrong for the constraint pointer model
    # Note: the name 'target_prefix' should be changed to 'constraints'
    use_constraint_pointer_model = kwargs.get('use_constraint_pointer_model', False)
    if use_constraint_pointer_model:
        logger.info('Using constraint pointer model datastream')
        stream = Mapping(stream,
                         ConstraintModelStreamTransformer(
                             sample_ratio=kwargs.get('train_sample_ratio', 1.),
                             min_suffix_source_ratio=kwargs.get('min_suffix_source_ratio', None),
                             nmt_baseline_training=kwargs.get('nmt_baseline_training', False)),
                         add_sources=('target_prefix', 'target_suffix', 'model_choice_sequence'))

    else:
        logger.info('Using default prefix IMT datastream')
        stream = Mapping(stream,
                         PrefixSuffixStreamTransformer(
                             sample_ratio=kwargs.get('train_sample_ratio', 1.),
                             min_suffix_source_ratio=kwargs.get('min_suffix_source_ratio', None),
                             nmt_baseline_training=kwargs.get('nmt_baseline_training', False)),
                         add_sources=('target_prefix', 'target_suffix'))

    stream = Mapping(stream, CopySourceAndTargetToMatchPrefixes(stream))

    # changing stream.produces_examples is a little hack which lets us use Unpack to flatten
    stream.produces_examples = False
    # flatten the stream back out into (source, target, target_prefix, target_suffix)
    stream = Unpack(stream)

    # WORKING: Optionally use the source prefix transformer to create a stream for the constraint model
    # WORKING: whether to use this transformer depends upon configuration
    # TODO: rename 'use_constraint_model' config to something more informative
    if kwargs.get('use_constraint_model', False):
        begin_constraint_idx = src_vocab[kwargs['begin_constraint_token']]
        end_constraint_idx = src_vocab[kwargs['end_constraint_token']]
        gap_idx = trg_vocab[kwargs['output_gap_token']]
        stream = Mapping(stream, SourceAndPrefixTransformer(begin_constraint_idx, end_constraint_idx, gap_idx))

        # e = stream.get_epoch_iterator()
        # t = e.next()
        # import ipdb; ipdb.set_trace()

    # Now make a very big batch that we can shuffle
    shuffle_batch_size = kwargs['shuffle_batch_size']
    stream = Batch(stream, iteration_scheme=ConstantScheme(shuffle_batch_size))

    # shuffle the big batches
    stream = ShuffleBatchTransformer(stream)

    # unpack it again
    stream = Unpack(stream)

    # Build a batched version of stream to read k batches ahead
    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size * sort_k_batches))

    # Sort all samples in the read-ahead batch
    stream = Mapping(stream, SortMapping(_length))

    # Convert it into a stream again
    stream = Unpack(stream)

    # Finally, construct batches from the stream with specified batch size
    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size))

    # Pad sequences that are short
    # TODO: is it correct to blindly pad the target_prefix and the target_suffix?
    configurable_padding_args = {
        'suffix_length': kwargs.get('suffix_length', None),
        'truncate_sources': kwargs.get('truncate_sources', [])
    }
    mask_sources = ('source', 'target', 'target_prefix', 'target_suffix')
    padding_symbols = [src_vocab_size - 1, trg_vocab_size - 1, trg_vocab_size - 1, trg_vocab_size - 1]
    # TODO: Model choice sequence actually doesn't need to be masked??
    if use_constraint_pointer_model:
        mask_sources = mask_sources + ('model_choice_sequence',)
        # we pad the model choice sequence with 0
        padding_symbols.append(0)

    logger.info('Training suffix length is: {}'.format(configurable_padding_args['suffix_length']))
    logger.info('I will mask the following sources after <suffix_length>: {}'.format(configurable_padding_args['truncate_sources']))
    masked_stream = PaddingWithEOS(stream, padding_symbols, mask_sources=mask_sources, **configurable_padding_args)

    return masked_stream, src_vocab, trg_vocab


# # Remember that the BleuValidator does hackish stuff to get target set information from the main_loop data_stream
# # using all kwargs here makes it more clear that this function is always called with get_dev_stream(**config_dict)
def get_dev_stream_with_prefixes(val_set=None, val_set_grndtruth=None, src_vocab=None, src_vocab_size=30000,
                                 trg_vocab=None, trg_vocab_size=30000, unk_id=1, return_vocab=False, use_constraint_pointer_model=False, **kwargs):
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
                                      bos_token=u'<S>',
                                      eos_token=u'</S>',
                                      unk_token=u'<UNK>',
                                      encoding='utf8')
        dev_target_dataset = TextFile([val_set_grndtruth], trg_vocab,
                                      bos_token=u'<S>',
                                      eos_token=u'</S>',
                                      unk_token=u'<UNK>',
                                      encoding='utf8')

        dev_stream = Merge([dev_source_dataset.get_example_stream(),
                            dev_target_dataset.get_example_stream()],
                           ('source', 'target'))

        # now add prefix and suffixes to this stream
        if use_constraint_pointer_model:
            dev_stream = Mapping(dev_stream, ConstraintModelStreamTransformer(sample_ratio=kwargs.get('dev_sample_ratio', 1.)),
                                 add_sources=('target_prefix', 'target_suffix', 'model_choice_sequence'))
        else:
            dev_stream = Mapping(dev_stream, PrefixSuffixStreamTransformer(sample_ratio=kwargs.get('dev_sample_ratio', 1.)),
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


# WORKING: same as get_dev_stream_with_prefixes, but user provides the prefix file directly
def get_dev_stream_with_prefix_file(val_set=None, val_set_grndtruth=None, val_set_prefixes=None, val_set_suffixes=None,
                                    src_vocab=None, src_vocab_size=30000, trg_vocab=None, trg_vocab_size=30000, unk_id=1,
                                    return_vocab=False, **kwargs):
    """Setup development stream with user-provided source, target, prefixes, and suffixes"""

    dev_stream = None
    if val_set is not None and val_set_grndtruth is not None and val_set_prefixes is not None and val_set_suffixes is not None:
        src_vocab = _ensure_special_tokens(
            src_vocab if isinstance(src_vocab, dict) else
            cPickle.load(open(src_vocab)),
            bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)

        trg_vocab = _ensure_special_tokens(
            trg_vocab if isinstance(trg_vocab, dict) else
            cPickle.load(open(trg_vocab)),
            bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)

        # Note: user should have already provided the EOS token in the data representation for the suffix
        # Note: The reason that we need EOS tokens in the reference file is that IMT systems need to evaluate metrics
        # Note: which count prediction of the </S> token, and evaluation scripts are called on the files
        dev_source_dataset = TextFile([val_set], src_vocab,
                                      bos_token=u'<S>',
                                      eos_token=u'</S>',
                                      unk_token=u'<UNK>',
                                      encoding='utf8')
        dev_target_dataset = TextFile([val_set_grndtruth], trg_vocab,
                                      bos_token=u'<S>',
                                      eos_token=u'</S>',
                                      unk_token=u'<UNK>',
                                      encoding='utf8')
        dev_prefix_dataset = TextFile([val_set_prefixes], trg_vocab,
                                      bos_token=u'<S>',
                                      eos_token=None,
                                      unk_token=u'<UNK>',
                                      encoding='utf8')
        dev_suffix_dataset = TextFile([val_set_suffixes], trg_vocab,
                                      bos_token=None,
                                      eos_token=None,
                                      unk_token=u'<UNK>',
                                      encoding='utf8')

        dev_stream = Merge([dev_source_dataset.get_example_stream(),
                            dev_target_dataset.get_example_stream(),
                            dev_prefix_dataset.get_example_stream(),
                            dev_suffix_dataset.get_example_stream()],
                           ('source', 'target','target_prefix','target_suffix'))

    if return_vocab:
        return dev_stream, src_vocab, trg_vocab
    else:
        return dev_stream

class CopySourceAndPrefixNTimes(Transformer):
    """Duplicate the source N times to match the number of samples

    We need this transformer because the attention model expects one source sequence for each
    target sequence, but in the sampling case there are effectively (instances*sample_size) target sequences

    Parameters
    ----------
    data_stream : :class:`AbstractDataStream` instance
        The data stream to wrap
    n_samples : int -- the number of samples that were generated for each source sequence

    """
    def __init__(self, data_stream, n_samples=5, **kwargs):
        if data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches of '
                             'examples, not examples')
        self.n_samples = n_samples

        super(CopySourceAndPrefixNTimes, self).__init__(
            data_stream, produces_examples=False, **kwargs)


    @property
    def sources(self):
        return self.data_stream.sources

    def transform_batch(self, batch):
        batch_with_expanded_source = []
        for i, (source, source_batch) in enumerate(
                zip(self.data_stream.sources, batch)):
            if source == 'source' or source == 'target_prefix':

                # copy each source sequence self.n_samples times, but keep the tensor 2d
                expanded_source = []
                for ins in source_batch:
                    expanded_source.extend([ins for _ in range(self.n_samples)])

                batch_with_expanded_source.append(expanded_source)
            else:
                batch_with_expanded_source.append(source_batch)

        return tuple(batch_with_expanded_source)


# WORKING: filter which removes instances with only very good or only very bad samples
# note that our scores are 1-metric, so a very high score is very bad
def filter_by_sample_score(data_tuple, max_score=0.98, min_score=0.1):
    """Assumes scores are the last element in the datastream tuple."""

    scores = data_tuple[-1]
    avg_score = numpy.mean(scores)
    if avg_score > max_score or avg_score < min_score:
        return False
    return True

