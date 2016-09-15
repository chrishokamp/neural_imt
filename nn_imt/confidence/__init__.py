"""
The confidence model asks how sure we are that our next prediction will be correct

Training input for this model is (source, prefix, CORRECT \in {0,1}), where the 1/0 label
is obtained by comparing the prediction of the current IMT model to the reference.

The model outputs a single scalar which can be interpreted as p(correct(w_t))

IMPLEMENTATION:
Pass the initial states through the Readout to get the prediction

The sequence generator for this model is the same one we are using to generate the data

The cost function is binary crossentropy for (batch, prediction) -- i.e (batch, 1)

binary cross entropy http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#theano.tensor.nnet.nnet.binary_crossentropy

At inference time, if we pass along the state for every entry in the beam search, we can also get the model's confidence at each time step

"""

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

from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)
from fuel.schemes import ConstantScheme

from blocks.algorithms import (GradientDescent, StepClipping,
                               CompositeRule, Adam, AdaDelta, Scale, RemoveNotFinite)
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.select import Selector
from blocks_extras.extensions.plot import Plot

from machine_translation.checkpoint import CheckpointNMT, LoadNMT, RunExternalValidation
from machine_translation.model import BidirectionalEncoder

from machine_translation.stream import _ensure_special_tokens

from nn_imt.sample import BleuValidator, Sampler, SamplingBase, IMT_F1_Validator
from nn_imt.model import NMTPrefixDecoder
from nn_imt.min_risk import get_sampling_model_and_input
from nn_imt.sample import SampleFunc
from nn_imt.stream import (PrefixSuffixStreamTransformer, CopySourceAndTargetToMatchPrefixes,
                           IMTSampleStreamTransformer, CopySourceAndPrefixNTimes, _length,
                           filter_by_sample_score, CallFunctionOnStream)
from nn_imt import load_params_and_get_beam_search

from machine_translation.stream import (get_textfile_stream, _too_long, _oov_to_unk,
                                        ShuffleBatchTransformer, PaddingWithEOS, FlattenSamples)

from nn_imt.stream import map_pair_to_imt_triples, get_dev_stream_with_prefixes
from nn_imt.evaluation import imt_f1
from nn_imt.search import BeamSearch

try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

logger = logging.getLogger(__name__)



# Note: this is to create the sampling model -- the cost model gets built from the output of this function
# This model will be used to check whether the prediction is right or wrong at each timestep, this is the labeling
# training data for the confidence model
def get_prediction_function(exp_config):

    # Create Theano variables
    logger.info('Creating theano variables')
    source_sentence = tensor.lmatrix('source')
    source_sentence_mask = tensor.matrix('source_mask')
    target_sentence = tensor.lmatrix('target_suffix')
    target_sentence_mask = tensor.matrix('target_suffix_mask')
    target_prefix = tensor.lmatrix('target_prefix')
    target_prefix_mask = tensor.matrix('target_prefix_mask')

    # build the model
    encoder = BidirectionalEncoder(
        exp_config['src_vocab_size'], exp_config['enc_embed'], exp_config['enc_nhids'])

    # Note: the 'min_risk' kwarg tells the decoder which sequence_generator and cost_function to use
    decoder = NMTPrefixDecoder(
        exp_config['trg_vocab_size'], exp_config['dec_embed'], exp_config['dec_nhids'],
        exp_config['enc_nhids'] * 2, loss_function='cross_entropy')

    # rename to match baseline NMT systems
    decoder.name = 'decoder'

    prediction_tags = decoder.prediction_tags(
        encoder.apply(source_sentence, source_sentence_mask),
        source_sentence_mask, target_sentence, target_sentence_mask,
        target_prefix, target_prefix_mask)

    logger.info('Creating computational graph')

    prediction_model = Model(prediction_tags)

    # Note that the parameters of this model must be pretrained, otherwise this doesn't make sense
    param_values = LoadNMT.load_parameter_values(exp_config['saved_parameters'], brick_delimiter=None)
    LoadNMT.set_model_parameters(prediction_model, param_values)

    prediction_function = prediction_model.get_theano_function()

    return prediction_function


# Copied from IMT main loop -- hacked version to do confidence prediction output
def main(config, tr_stream, dev_stream, source_vocab, target_vocab, use_bokeh=False):

    # add the tags from this function to the IMT datastream
    # prediction function signature
    # [target_suffix, source_mask, source, target_prefix_mask, target_prefix, target_suffix_mask]
    prediction_function = get_prediction_function(exp_config=config)

    tr_stream = Mapping(tr_stream, CallFunctionOnStream(prediction_function, [1, 0, 5, 4, 7, 6]),
    #tr_stream = Mapping(tr_stream, CallFunctionOnStream(prediction_function, [6, 1, 0, 5, 4, 7]),
                        add_sources=('readouts', 'prediction_tags'))

    import ipdb; ipdb.set_trace()

    # Create the prediction confidence model
    # the first draft of this model uses the readout output (before the post-merge step) as the per-timestep state vector

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

    # WORKING HERE:
    # symbolic variable which tags each timestep as GOOD/BAD
    # Note: later this might be tags for a hypothesis, right now the timesteps are actually determined by the reference
    # By zipping the confidence model output with the reference, we get the model's confidence that this reference word
    # will be predicted correctly
    prediction_tags = tensor.matrix('prediction_tags')
    readouts = tensor.tensor3('readouts')

    # Construct model
    logger.info('Building RNN encoder-decoder')
    encoder = BidirectionalEncoder(
        config['src_vocab_size'], config['enc_embed'], config['enc_nhids'])

    decoder = NMTPrefixDecoder(
        config['trg_vocab_size'], config['dec_embed'], config['dec_nhids'],
        config['enc_nhids'] * 2, loss_function='cross_entropy')

    # rename to match baseline NMT systems
    decoder.name = 'decoder'

    cost = decoder.confidence_cost(
        encoder.apply(source_sentence, source_sentence_mask),
        source_sentence_mask, target_sentence, target_sentence_mask,
        target_prefix, target_prefix_mask, readouts, prediction_tags)

    logger.info('Creating computational graph')
    # working: implement cost for confidence model
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

    import ipdb;ipdb.set_trace()

    # apply dropout for regularization
    if config['dropout'] < 1.0:
        # dropout is applied to the output of maxout in ghog
        # this is the probability of dropping out, so you probably want to make it <=0.5
        logger.info('Applying dropout')
        dropout_inputs = [x for x in cg.intermediary_variables
                          if x.name == 'maxout_apply_output']
        cg = apply_dropout(cg, dropout_inputs, config['dropout'])

    # WORKING: implement confidence -- remove all params except output model
    cost_model = Model(cost)

    model_params = cost_model.get_parameter_dict()
    trainable_params = cg.parameters
    import ipdb;ipdb.set_trace()
    print('trainable params')
    #params_to_remove = [model_params[k] for k in model_params.keys() if 'confidence' not in k]
    #for p in params_to_remove:
    #    trainable_params.remove(p)

    # target_embeddings = model.get_parameter_dict()['/target_recurrent_lm_with_alignments/target_embeddings.W']
    # trainable_params.remove(source_embeddings)
    # trainable_params.remove(target_embeddings)
    # END WORKING: implement confidence -- remove all params except output model



    # TODO: fixed dropout mask for recurrent params?
    # Print shapes
    # shapes = [param.get_value().shape for param in cg.parameters]
    # logger.info("Parameter shapes: ")
    # for shape, count in Counter(shapes).most_common():
    #     logger.info('    {:15}: {}'.format(shape, count))
    # logger.info("Total number of parameters: {}".format(len(shapes)))

    # Print parameter names
    # enc_dec_param_dict = merge(Selector(encoder).get_parameters(),
    #                            Selector(decoder).get_parameters())
    # logger.info("Parameter names: ")
    # for name, value in enc_dec_param_dict.items():
    #     logger.info('    {:15}: {}'.format(value.get_value().shape, name))
    # logger.info("Total number of parameters: {}"
    #             .format(len(enc_dec_param_dict)))

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
        # TrainingDataMonitoring(trainable_params, after_batch=True),
        Printing(after_batch=True),
        CheckpointNMT(config['saveto'],
                      every_n_batches=config['save_freq'])
    ]

    # WORKING: confidence prediction
    #monitor everything that could possibly be relevant


    # Set up the sampling graph for validation during training
    # Theano variables for the sampling graph
    # Note this also loads the model parameters
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
    if config['bleu_script'] is not None:
        logger.info("Building bleu validator")
        extensions.append(
            BleuValidator(sampling_input, sampling_prefix, samples=samples, config=config,
                          model=search_model, data_stream=dev_stream,
                          src_vocab=source_vocab,
                          trg_vocab=target_vocab,
                          normalize=config['normalized_bleu'],
                          every_n_batches=config['bleu_val_freq']))

    # TODO: add first-word accuracy validation
    # TODO: add IMT meteor early stopping
    if config.get('imt_f1_validation', None) is not None:
        logger.info("Building imt F1 validator")
        extensions.append(
            IMT_F1_Validator(sampling_input, sampling_prefix,
                             samples=samples,
                             config=config,
                             model=search_model, data_stream=dev_stream,
                             src_vocab=source_vocab,
                             trg_vocab=target_vocab,
                             normalize=config['normalized_bleu'],
                             every_n_batches=config['bleu_val_freq']))

    # Reload model if necessary
    # if config['reload']:
    #     extensions.append(LoadNMT(config['saveto']))

    # Plot cost in bokeh if necessary
    if use_bokeh and BOKEH_AVAILABLE:
        extensions.append(
            Plot(config['model_save_directory'], channels=[['decoder_cost_cost'], ['validation_set_bleu_score', 'validation_set_imt_f1_score']],
                 every_n_batches=10))

    # Set up training algorithm
    logger.info("Initializing training algorithm")

    # WORKING: implement confidence model
    # if there is dropout or random noise, we need to use the output of the modified graph
    algorithm = GradientDescent(
        cost=cg.outputs[0], parameters=trainable_params,
        step_rule=CompositeRule([StepClipping(config['step_clipping']),
                          eval(config['step_rule'])(), RemoveNotFinite()]),
        # step_rule=CompositeRule([StepClipping(10.0), Scale(0.01)]),
        on_unused_sources='warn'
    )
    #if config['dropout'] < 1.0:
    #   algorithm = GradientDescent(
    #       cost=cg.outputs[0], parameters=trainable_params,
    #       step_rule=CompositeRule([StepClipping(config['step_clipping']),
    #                         eval(config['step_rule'])(), RemoveNotFinite()]),
    #       # step_rule=CompositeRule([StepClipping(10.0), Scale(0.01)]),
    #       on_unused_sources='warn'
    #   )
    #else:
    #   algorithm = GradientDescent(
    #       cost=cost, parameters=cg.parameters,
    #       step_rule=CompositeRule([StepClipping(config['step_clipping']),
    #                                eval(config['step_rule'])()]),
    #       on_unused_sources='warn'
    #   )
    # END WORKING: implement confidence model

    import ipdb;ipdb.set_trace()


    # enrich the logged information
    extensions.append(
        Timing(every_n_batches=100)
    )


    # WORKING: debugging confidence
    # get theano function from model
    # WORKING: implement word-level confidence cost
    #   @application(inputs=['representation', 'source_sentence_mask',
    #                                'target_sentence_mask', 'target_sentence', 'target_prefix_mask', 'target_prefix'],
    #                                                 outputs=['cost'])
    #       def confidence_cost(self, representation, source_sentence_mask,
    #                            target_sentence, target_sentence_mask, target_prefix, target_prefix_mask):



    logger.info('Creating theano variables')
    #source_sentence = tensor.lmatrix('source')
    #source_sentence_mask = tensor.matrix('source_mask')

    # Note that the _names_ are changed from normal NMT
    # for IMT training, we use only the suffix as the reference
    #target_sentence = tensor.lmatrix('target_suffix')
    #target_sentence_mask = tensor.matrix('target_suffix_mask')
    # TODO: change names back to *_suffix, there is currently a theano function name error
    # TODO: in the GradientDescent Algorithm

    #target_prefix = tensor.lmatrix('target_prefix')
    #target_prefix_mask = tensor.matrix('target_prefix_mask')


    # confidence_output = decoder.confidence_cost(
    #     encoder.apply(source_sentence, source_sentence_mask),
    #     source_sentence_mask, target_sentence, target_sentence_mask,
    #     target_prefix, target_prefix_mask)

    # confidence_model = Model(confidence_output)

    # t_cost_func = confidence_model.get_theano_function()
    # inputs
    # [source_mask, source, target_prefix_mask, target_prefix, target_suffix_mask, target_suffix]

    #import ipdb;ipdb.set_trace()

    # get the right args from the datastream
    # TODO: just print source, prefix, suffix, prediction, correct to new files -- this makes sure everything is aligned
    # OUTPUT_DIR = '/media/1tb_drive/imt_models/word_prediction_accuracy_experiments/en-de/exp_1'
    # for the_file in os.listdir(OUTPUT_DIR):
    #     file_path = os.path.join(OUTPUT_DIR, the_file)
    #     try:
    #         if os.path.isfile(file_path):
    #             os.unlink(file_path)
    #     except Exception as e:
    #         print(e)
    #
    # def write_file_truncate_mask(filename, data, mask, mode='a'):
    #     ''' data is list of list '''
    #
    #     assert len(data) == len(mask)
    #     with codecs.open(filename, mode, encoding='utf8') as out:
    #         for l, m in zip(data, mask):
    #             output = u' '.join(l[:int(m.sum())]) + u'\n'
    #             out.write(output)
    #     logger.info('Wrote file: {}'.format(filename))
    #
    #
    # target_ivocab = {k:v.decode('utf8') for v,k in target_vocab.items()}
    # source_ivocab = {k:v.decode('utf8') for v,k in source_vocab.items()}
    # import ipdb; ipdb.set_trace()
    # tag_ivocab = {1: 'True', 0: 'False'}
    #
    # test_iter = tr_stream.get_epoch_iterator()
    # it = 0
    # for t_source, t_source_mask, t_target, t_target_mask, t_target_prefix, t_target_prefix_mask, t_target_suffix, t_target_suffix_mask in test_iter:
    #     if it <= 1000:
    #         it += 1
    #         t_cost = t_cost_func(t_source_mask, t_source, t_target_prefix_mask, t_target_prefix, t_target_suffix_mask, t_target_suffix)
    #         readouts = t_cost[0]
    #         preds = readouts.argmax(axis=2)
    #         correct = preds.T == t_target_suffix
    #
    #
    #         source_output = os.path.join(OUTPUT_DIR,'sources.en')
    #         prefix_output = os.path.join(OUTPUT_DIR,'prefixes.de')
    #         suffix_output = os.path.join(OUTPUT_DIR,'suffixes.de')
    #         prediction_output = os.path.join(OUTPUT_DIR,'predictions.de')
    #         correct_output = os.path.join(OUTPUT_DIR,'prefix_word_prediction_acc.out')
    #
    #         source_text = [[source_ivocab[w] for w in s] for s in t_source]
    #         prefix_text = [[target_ivocab[w] for w in s] for s in t_target_prefix]
    #         suffix_text = [[target_ivocab[w] for w in s] for s in t_target_suffix]
    #         pred_text = [[target_ivocab[w] for w in s] for s in preds.T]
    #         correct_text = [[tag_ivocab[w] for w in s] for s in correct]
    #
    #
    #         for triple in zip([source_output, prefix_output, suffix_output, prediction_output, correct_output],
    #                           [source_text, prefix_text, suffix_text, pred_text, correct_text],
    #                           [t_source_mask, t_target_prefix_mask, t_target_suffix_mask, t_target_suffix_mask, t_target_suffix_mask]):
    #             write_file_truncate_mask(*triple)
    #     else:
    #         break
    #
    # import ipdb; ipdb.set_trace()

    #t_cost = t_cost_func(t_source, t_target_prefix)
    #t_cost = t_cost_func(t_target_suffix, t_source_mask, t_source, t_target_prefix_mask, t_target_prefix, t_target_suffix_mask)
    #t_cost = t_cost_func(t_source_mask, t_source, t_target_prefix_mask, t_target_prefix, t_target_suffix_mask, t_target_suffix)

    #    return confidence_cost, flat_y, confidence_logits, readouts


    #predictions = t_cost[0].argmax(axis=2)

    # TODO: next step -- print gradients and weights during training find out where nan is coming from
    # TODO: look at the gradient of this function with respect to parameters? -- see here: http://deeplearning.net/software/theano/tutorial/gradients.html

    # TODO: function which adds right/wrong tags for model predictions to the datastream. In this case we can learn a simple linear model as a baseline
    # TODO: print predictions for each batch for each timestep to file -- _dont shuffle_ so that we get the right order

    # import ipdb;ipdb.set_trace()


    # from blocks reverse_words example
    # observables = [
    #     cost, min_energy, max_energy, mean_activation,
    #     batch_size, max_length, cost_per_character,
    #     algorithm.total_step_norm, algorithm.total_gradient_norm]
    # for name, parameter in trainable_params.items():
    #     observables.append(parameter.norm(2).copy(name + "_norm"))
    #     observables.append(algorithm.gradients[parameter].norm(2).copy(
    #         name + "_grad_norm"))

    for i, (k,v) in enumerate(algorithm.updates):
        v.name = k.name + '_{}'.format(i)

    aux_vars = [v for v in cg.auxiliary_variables[-3:]]
    # import ipdb; ipdb.set_trace()



    extensions.extend([
        TrainingDataMonitoring([cost], after_batch=True),
        # TrainingDataMonitoring([v for k,v in algorithm.updates[:2]], after_batch=True),
        # TrainingDataMonitoring(aux_vars, after_batch=True),
        # TrainingDataMonitoring(trainable_params, after_batch=True),
        Printing(after_batch=True)]
    )

    # Initialize main loop
    logger.info("Initializing main loop")
    main_loop = MainLoop(
        model=training_model,
        algorithm=algorithm,
        data_stream=tr_stream,
        extensions=extensions
    )
    import ipdb;ipdb.set_trace()

    # Train!
    main_loop.run()


