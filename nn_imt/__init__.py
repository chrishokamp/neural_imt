import logging

import os
import shutil
from collections import Counter
from theano import tensor
from toolz import merge
import numpy
import pickle
from subprocess import Popen, PIPE
import codecs

from blocks.algorithms import (GradientDescent, StepClipping,
                               CompositeRule, Adam, AdaDelta)
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.select import Selector
from blocks.search import BeamSearch
from blocks_extras.extensions.plot import Plot

from machine_translation.checkpoint import CheckpointNMT, LoadNMT
from machine_translation.model import BidirectionalEncoder

from machine_translation.stream import _ensure_special_tokens

from nn_imt.sample import BleuValidator, Sampler, SamplingBase
from nn_imt.model import NMTPrefixDecoder
from nn_imt.stream import map_pair_to_imt_triples, get_dev_stream_with_prefixes
from nn_imt.evaluation import imt_f1

try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

logger = logging.getLogger(__name__)


# WORKING: train IMT with validation using BLEU/METEOR on suffix
# WORKING: remember to cut the suffix for validation to only one EOS token
def main(config, tr_stream, dev_stream, source_vocab, target_vocab, use_bokeh=False):

    # Create Theano variables
    logger.info('Creating theano variables')
    source_sentence = tensor.lmatrix('source')
    source_sentence_mask = tensor.matrix('source_mask')

    # Note that the _names_ are changed from normal NMT
    # for IMT training, we use only the suffix as the reference
    target_sentence = tensor.lmatrix('target_suffix')
    target_sentence_mask = tensor.matrix('target_suffix_mask')
    # TODO: change names back to *_suffix, there is currently a theano function name error
    # TODO: in the GradientDescent Algorithm

    target_prefix = tensor.lmatrix('target_prefix')
    target_prefix_mask = tensor.matrix('target_prefix_mask')


    # Construct model
    logger.info('Building RNN encoder-decoder')
    encoder = BidirectionalEncoder(
        config['src_vocab_size'], config['enc_embed'], config['enc_nhids'])

    decoder = NMTPrefixDecoder(
        config['trg_vocab_size'], config['dec_embed'], config['dec_nhids'],
        config['enc_nhids'] * 2, loss_function='cross_entropy')

    # rename to match baseline NMT systems
    decoder.name = 'decoder'

    # TODO: change the name of `target_sentence` to `target_suffix` for more clarity
    # cost = decoder.cost(
    #     encoder.apply(source_sentence, source_sentence_mask),
    #     source_sentence_mask, target_sentence, target_sentence_mask,
    #     target_prefix, target_prefix_mask)
    cost = decoder.cost(
        encoder.apply(source_sentence, source_sentence_mask),
        source_sentence_mask, target_sentence, target_sentence_mask,
        target_prefix, target_prefix_mask)

    logger.info('Creating computational graph')
    cg = ComputationGraph(cost)

    # INITIALIZATION
    logger.info('Initializing model')
    encoder.weights_init = decoder.weights_init = IsotropicGaussian(
        config['weight_scale'])
    encoder.biases_init = decoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    decoder.push_initialization_config()
    encoder.bidir.prototype.weights_init = Orthogonal()
    decoder.transition.weights_init = Orthogonal()
    encoder.initialize()
    decoder.initialize()

    # apply dropout for regularization
    if config['dropout'] < 1.0:
        # dropout is applied to the output of maxout in ghog
        # this is the probability of dropping out, so you probably want to make it <=0.5
        logger.info('Applying dropout')
        dropout_inputs = [x for x in cg.intermediary_variables
                          if x.name == 'maxout_apply_output']
        cg = apply_dropout(cg, dropout_inputs, config['dropout'])

    # Apply weight noise for regularization
    if config.get('weight_noise_ff', None) > 0.0:
        logger.info('Applying weight noise to ff layers')
        enc_params = Selector(encoder.lookup).get_parameters().values()
        enc_params += Selector(encoder.fwd_fork).get_parameters().values()
        enc_params += Selector(encoder.back_fork).get_parameters().values()
        dec_params = Selector(
            decoder.sequence_generator.readout).get_parameters().values()
        dec_params += Selector(
            decoder.sequence_generator.fork).get_parameters().values()
        dec_params += Selector(decoder.transition.initial_transformer).get_parameters().values()
        cg = apply_noise(cg, enc_params+dec_params, config['weight_noise_ff'])

    # TODO: weight noise for recurrent params isn't currently implemented -- see config['weight_noise_rec']
    # TODO: fixed dropout mask for recurrent params?

    # Print shapes
    shapes = [param.get_value().shape for param in cg.parameters]
    logger.info("Parameter shapes: ")
    for shape, count in Counter(shapes).most_common():
        logger.info('    {:15}: {}'.format(shape, count))
    logger.info("Total number of parameters: {}".format(len(shapes)))

    # Print parameter names
    enc_dec_param_dict = merge(Selector(encoder).get_parameters(),
                               Selector(decoder).get_parameters())
    logger.info("Parameter names: ")
    for name, value in enc_dec_param_dict.items():
        logger.info('    {:15}: {}'.format(value.get_value().shape, name))
    logger.info("Total number of parameters: {}"
                .format(len(enc_dec_param_dict)))

    # Set up training model
    logger.info("Building model")
    training_model = Model(cost)

    # create the training directory, and copy this config there if directory doesn't exist
    if not os.path.isdir(config['saveto']):
        os.makedirs(config['saveto'])
        shutil.copy(config['config_file'], config['saveto'])

    # Set extensions
    logger.info("Initializing extensions")
    extensions = [
        FinishAfter(after_n_batches=config['finish_after']),
        TrainingDataMonitoring([cost], after_batch=True),
        Printing(after_batch=True),
        CheckpointNMT(config['saveto'],
                      every_n_batches=config['save_freq'])
    ]

    # Set up the sampling graph for validation during training
    # Theano variables for the sampling graph
    sampling_vars = load_params_and_get_beam_search(config, encoder=encoder, decoder=decoder)
    beam_search, search_model, samples, sampling_input, sampling_prefix = sampling_vars

    if config['hook_samples'] >= 1:
        logger.info("Building sampler")
        extensions.append(
            Sampler(model=search_model, data_stream=tr_stream,
                    hook_samples=config['hook_samples'],
                    every_n_batches=config['sampling_freq'],
                    src_vocab=source_vocab,
                    trg_vocab=target_vocab,
                    src_vocab_size=config['src_vocab_size']))

    # Add early stopping based on bleu
    # TODO: add IMT BLEU early stopping with prefix decoding
    if config['bleu_script'] is not None:
        logger.info("Building bleu validator")
        extensions.append(
            BleuValidator(sampling_input, sampling_prefix, samples=samples, config=config,
                          model=search_model, data_stream=dev_stream,
                          src_vocab=source_vocab,
                          trg_vocab=target_vocab,
                          normalize=config['normalized_bleu'],
                          every_n_batches=config['bleu_val_freq']))

    # TODO: add IMT meteor early stopping

    # Reload model if necessary
    if config['reload']:
        extensions.append(LoadNMT(config['saveto']))

    # Plot cost in bokeh if necessary
    if use_bokeh and BOKEH_AVAILABLE:
        extensions.append(
            Plot(config['model_save_directory'], channels=[['decoder_cost_cost'], ['validation_set_bleu_score']],
                 every_n_batches=10))

    # Set up training algorithm
    logger.info("Initializing training algorithm")
    # if there is dropout or random noise, we need to use the output of the modified graph
    if config['dropout'] < 1.0 or config['weight_noise_ff'] > 0.0:
        algorithm = GradientDescent(
            cost=cg.outputs[0], parameters=cg.parameters,
            step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                     eval(config['step_rule'])()]),
            on_unused_sources='warn'
        )
    else:
        algorithm = GradientDescent(
            cost=cost, parameters=cg.parameters,
            step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                     eval(config['step_rule'])()]),
            on_unused_sources='warn'
        )

    # enrich the logged information
    extensions.append(
        Timing(every_n_batches=100)
    )

    # Initialize main loop
    logger.info("Initializing main loop")
    main_loop = MainLoop(
        model=training_model,
        algorithm=algorithm,
        data_stream=tr_stream,
        extensions=extensions
    )

    # Train!
    main_loop.run()


# TODO: use this function in training as well (at least for sampling and validation components)
# TODO: to get the pieces we need to setup the graphs
# TODO: break this function into parts
def load_params_and_get_beam_search(exp_config, decoder=None, encoder=None, brick_delimiter=None):

    if encoder is None:
        encoder = BidirectionalEncoder(
            exp_config['src_vocab_size'], exp_config['enc_embed'], exp_config['enc_nhids'])

    # Note: decoder should be None when we are just doing prediction, not validation
    if decoder is None:
        decoder = NMTPrefixDecoder(
            exp_config['trg_vocab_size'], exp_config['dec_embed'], exp_config['dec_nhids'],
            exp_config['enc_nhids'] * 2, loss_function='cross_entropy')
        # rename to match baseline NMT systems so that params can be transparently initialized
        decoder.name = 'decoder'

    # Create Theano variables
    logger.info('Creating theano variables')
    sampling_input = tensor.lmatrix('sampling_input')
    sampling_prefix = tensor.lmatrix('sampling_target_prefix')

    # Get beam search
    logger.info("Building sampling model")
    sampling_representation = encoder.apply(sampling_input, tensor.ones(sampling_input.shape))

    # Note: prefix can be empty if we want to simulate baseline NMT
    generated = decoder.generate(sampling_input, sampling_representation,
                                 target_prefix=sampling_prefix)

    # create the 1-step sampling graph
    _, samples = VariableFilter(
        bricks=[decoder.sequence_generator], name="outputs")(
        ComputationGraph(generated[1]))  # generated[1] is next_outputs

    # set up beam search
    beam_search = BeamSearch(samples=samples)

    logger.info("Creating Search Model...")
    search_model = Model(generated)

    # optionally set beam search model parameter values from an .npz file
    # Note: we generally would set the model params in this way when doing only prediction/evaluation
    if exp_config.get('load_from_saved_parameters', False):
        logger.info("Loading parameters from model: {}".format(exp_config['saved_parameters']))
        param_values = LoadNMT.load_parameter_values(exp_config['saved_parameters'], brick_delimiter=brick_delimiter)
        LoadNMT.set_model_parameters(search_model, param_values)

    return beam_search, search_model, samples, sampling_input, sampling_prefix

# TODO: does the predictor need modes? -- prefix prediction vs prediction from scratch
class IMTPredictor:
    """"Uses a trained NMT model to do IMT prediction -- prediction where input includes a prefix"""

    sutils = SamplingBase()

    def __init__(self, exp_config):

        theano_variables = load_params_and_get_beam_search(exp_config,
                                                           brick_delimiter=exp_config.get('brick_delimiter', None))
        # beam_search, search_model, samples, sampling_input, sampling_prefix = sampling_vars
        self.beam_search, search_model, samples, self.source_sampling_input, self.target_sampling_input = theano_variables

        self.exp_config = exp_config
        # how many hyps should be output (only used in file prediction mode)
        self.n_best = exp_config.get('n_best', 1)

        self.source_lang = exp_config.get('source_lang', 'en')
        self.target_lang = exp_config.get('target_lang', 'es')

        tokenize_script = exp_config.get('tokenize_script', None)
        detokenize_script = exp_config.get('detokenize_script', None)
        if tokenize_script is not None and detokenize_script is not None:
            self.source_tokenizer_cmd = [tokenize_script, '-l', self.source_lang, '-q', '-', '-no-escape', '1']
            self.target_tokenizer_cmd = [tokenize_script, '-l', self.target_lang, '-q', '-', '-no-escape', '1']
            self.detokenizer_cmd = [detokenize_script, '-l', self.target_lang, '-q', '-']
        else:
            self.source_tokenizer_cmd = None
            self.target_tokenizer_cmd = None
            self.detokenizer_cmd = None

        # this index will get overwritten with the EOS token by _ensure_special_tokens
        # IMPORTANT: the index must be created in the same way it was for training,
        # otherwise the predicted indices will be nonsense
        # Make sure that src_vocab_size and trg_vocab_size are correct in your configuration
        self.src_eos_idx = exp_config['src_vocab_size'] - 1
        self.trg_eos_idx = exp_config['trg_vocab_size'] - 1

        self.unk_idx = exp_config['unk_id']

        # Get vocabularies and inverse indices
        self.src_vocab = _ensure_special_tokens(
            pickle.load(open(exp_config['src_vocab'])), bos_idx=0,
            eos_idx=self.src_eos_idx, unk_idx=self.unk_idx)
        self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        self.trg_vocab = _ensure_special_tokens(
            pickle.load(open(exp_config['trg_vocab'])), bos_idx=0,
            eos_idx=self.trg_eos_idx, unk_idx=self.unk_idx)
        self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}

        self.unk_idx = self.unk_idx

    def map_idx_or_unk(self, sentence, index, unknown_token='<UNK>'):
        if type(sentence) is str:
            sentence = sentence.split()
        return [index.get(w, unknown_token) for w in sentence]

    def predict_files(self, source_file, prefix_file, output_file=None):

        tokenize = self.source_tokenizer_cmd is not None
        detokenize = self.detokenizer_cmd is not None

        if output_file is not None:
            ftrans = codecs.open(output_file, 'wb', encoding='utf8')
        else:
            # cut off the language suffix to make output file name
            output_file = '.'.join(source_file.split('.')[:-1]) + '.trans.out'
            ftrans = codecs.open(output_file, 'wb', encoding='utf8')

        logger.info("Started translation, will output {} translations for each segment"
                    .format(self.n_best))
        total_cost = 0.0

        # WORKING:
        with codecs.open(source_file, encoding='utf8') as srcs:
            with codecs.open(prefix_file, encoding='utf8') as prefixes:

                # map (source, ref) to [(source, prefix, suffix)]
                source_lines = srcs.read().strip().split('\n')
                prefix_lines = prefixes.read().strip().split('\n')

                assert len(source_lines) == len(prefix_lines), 'Source and reference files must be the same length'
                for i, instance in enumerate(zip(source_lines, prefix_lines)):
                    logger.info("Translating segment: {}".format(i))

                    # Note: tokenization is not currently implemented -- assumes whitespace tokenization!!!
                    # Right now, tokenization happens in self.map_idx_or_unk if predict_segment is passed a string
                    source_seq = instance[0].split()
                    prefix_seq = instance[1].split()


                    translations, costs = self.predict_segment(source_seq, target_prefix=prefix_seq,
                                                               n_best=self.n_best,
                                                               tokenize=tokenize, detokenize=detokenize)

                    # predict_segment returns a list of hyps, we take the best ones
                    nbest_translations = translations[:self.n_best]
                    nbest_costs = costs[:self.n_best]

                    if self.n_best == 1:
                        ftrans.write((nbest_translations[0] + '\n').decode('utf8'))
                        total_cost += nbest_costs[0]
                    else:
                        # one blank line to separate each nbest list
                        ftrans.write('\n'.join(nbest_translations).decode('utf8') + '\n\n')
                        total_cost += sum(nbest_costs)

                    if i != 0 and i % 100 == 0:
                        logger.info("Translated {} lines of test set...".format(i))

        logger.info("Saved translated output to: {}".format(ftrans.name))
        logger.info("Total cost of the test: {}".format(total_cost))
        ftrans.close()

        return output_file

    def predict_segment(self, segment, target_prefix=None, n_best=1, tokenize=False, detokenize=False):
        """
        Do prediction for a single segment, which is a list of token idxs

        Parameters
        ----------
        segment: list[int] : a list of int indexes representing the input sequence in the source language
        n_best: int : how many hypotheses to return (must be <= beam_size)
        tokenize: bool : does the source segment need to be tokenized first?
        detokenize: bool : do the output hypotheses need to be detokenized?

        Returns
        -------
        trans_out: str : the best translation according to beam search
        cost: float : the cost of the best translation

        """

        if tokenize:
            source_tokenizer = Popen(self.source_tokenizer_cmd, stdin=PIPE, stdout=PIPE)
            segment, _ = source_tokenizer.communicate(segment)
            # if there is a prefix, we need to tokenize and preprocess it also
            if target_prefix is not None:
                target_tokenizer = Popen(self.target_tokenizer_cmd, stdin=PIPE, stdout=PIPE)
                target_prefix, _ = target_tokenizer.communicate(target_prefix)

        segment = self.map_idx_or_unk(segment, self.src_vocab, self.unk_idx)
        segment += [self.src_eos_idx]

        seq = IMTPredictor.sutils._oov_to_unk(
            segment, self.exp_config['src_vocab_size'], self.unk_idx)
        input_ = numpy.tile(seq, (self.exp_config['beam_size'], 1))

        # TODO: HANDLE THE CASE WHERE TARGET PREFIX IS EMPTY
        if target_prefix is not None:
            print('predicting target prefix: {}'.format(target_prefix))
            target_prefix = self.map_idx_or_unk(target_prefix, self.trg_vocab, self.unk_idx)
            prefix_seq = IMTPredictor.sutils._oov_to_unk(
                target_prefix, self.exp_config['trg_vocab_size'], self.unk_idx)

            prefix_input_ = numpy.tile(prefix_seq, (self.exp_config['beam_size'], 1))
            # draw sample, checking to ensure we don't get an empty string back
            trans, costs = \
                self.beam_search.search(
                    input_values={self.source_sampling_input: input_,
                                  self.target_sampling_input: prefix_input_},
                    max_length=3*len(seq), eol_symbol=self.trg_eos_idx,
                    ignore_first_eol=False)

        else:
            # draw sample, checking to ensure we don't get an empty string back
            trans, costs = \
                self.beam_search.search(
                    input_values={self.sampling_input: input_},
                    max_length=3*len(seq), eol_symbol=self.trg_eos_idx,
                    ignore_first_eol=False)

        # normalize costs according to the sequence lengths
        if self.exp_config['normalized_bleu']:
            lengths = numpy.array([len(s) for s in trans])
            costs = costs / lengths

        best_n_hyps = []
        best_n_costs = []
        best_n_idxs = numpy.argsort(costs)[:n_best]
        for j, idx in enumerate(best_n_idxs):
            try:
                trans_out_idxs = trans[idx]
                cost = costs[idx]

                # convert idx to words
                # `line` is a tuple with one item
                try:
                    assert trans_out_idxs[-1] == self.trg_eos_idx, 'Target hypothesis should end with the EOS symbol'
                    trans_out_idxs = trans_out_idxs[:-1]
                    src_in = IMTPredictor.sutils._idx_to_word(segment, self.src_ivocab)
                    trans_out = IMTPredictor.sutils._idx_to_word(trans_out_idxs, self.trg_ivocab)
                except AssertionError as e:
                    src_in = IMTPredictor.sutils._idx_to_word(segment, self.src_ivocab)
                    trans_out = IMTPredictor.sutils._idx_to_word(trans_out_idxs, self.trg_ivocab)
                    logger.error("ERROR: {} does not end with the EOS symbol".format(trans_out))
                    logger.error("I'm continuing anyway...")
            except ValueError:
                logger.info("Can NOT find a translation for line: {}".format(src_in))
                trans_out = '<UNK>'
                cost = 0.

            if detokenize:
                detokenizer = Popen(self.detokenizer_cmd, stdin=PIPE, stdout=PIPE)
                trans_out, _ = detokenizer.communicate(trans_out)
                # strip off the eol symbol
                trans_out = trans_out.strip()

            logger.info("Source: {}".format(src_in))
            logger.info("Target Hypothesis: {}".format(trans_out))

            best_n_hyps.append(trans_out)
            best_n_costs.append(cost)

        return best_n_hyps, best_n_costs


# TODO: use the refs properly as specified in the function signature
def split_refs_into_prefix_suffix_files(refs_file, config_obj, n_best=1):

    predict_stream, src_vocab, trg_vocab = get_dev_stream_with_prefixes(val_set=config_obj['test_set'],
                                                                        val_set_grndtruth=config_obj['test_gold_refs'],
                                                                        src_vocab=config_obj['src_vocab'],
                                                                        src_vocab_size=config_obj['src_vocab_size'],
                                                                        trg_vocab=config_obj['trg_vocab'],
                                                                        trg_vocab_size=config_obj['trg_vocab_size'],
                                                                        unk_id=config_obj['unk_id'],
                                                                        return_vocab=True)


    src_ivocab = {v: k for k, v in src_vocab.items()}
    trg_ivocab = {v: k for k, v in trg_vocab.items()}

    if not os.path.isdir(config_obj['model_save_directory']):
        os.mkdir(config_obj['model_save_directory'])

    # now create all of the prefixes and write them to a temporary file
    prediction_prefixes = os.path.join(config_obj['model_save_directory'], 'reference_prefixes.generated')
    prediction_suffixes = os.path.join(config_obj['model_save_directory'], 'reference_suffixes.generated')
    dup_sources_file = prediction_prefixes+'.sources'

    sampling_base = SamplingBase()
    # Note: we need to write a new file for sources as well, so that each source is duplicated
    # Note: the necessary number of times
    with codecs.open(dup_sources_file, 'w', encoding='utf8') as dup_sources:
        with codecs.open(prediction_prefixes, 'w', encoding='utf8') as prefix_file:
            with codecs.open(prediction_suffixes, 'w', encoding='utf8') as suffix_file:
                for l in list(predict_stream.get_epoch_iterator()):
                    # currently our datastream is (source,target,prefix,suffix)
                    source = l[0]
                    prefix = l[-2]
                    suffix = l[-1]
                    source_text = sampling_base._idx_to_word(source, src_ivocab)
                    prefix_text = sampling_base._idx_to_word(prefix, trg_ivocab)
                    suffix_text = sampling_base._idx_to_word(suffix, trg_ivocab)
                    assert len(prefix_text) > 0, 'prefix cannot be empty'

                    dup_sources.write(source_text.decode('utf8') + '\n')
                    prefix_file.write(prefix_text.decode('utf8') + '\n')

                    # we use the suffix file as references for the n-best list, so we may need to duplicate it
                    for _ in range(n_best):
                        suffix_file.write(suffix_text.decode('utf8') + '\n')
                    # if this is an n_best list, separate lists by new lines
                    if n_best > 1:
                        suffix_file.write('\n')


    return dup_sources_file, prediction_prefixes, prediction_suffixes


