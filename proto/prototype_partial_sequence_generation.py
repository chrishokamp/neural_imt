
# coding: utf-8


# make a decoder which uses the PartialSequenceGenerator to set up the initial states for sequence completion


import os
import codecs
import subprocess
from pprint import pprint
from subprocess import Popen, PIPE, STDOUT

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")


# In[ ]:

import numpy
import codecs
import tempfile
import cPickle
import os
import copy
from collections import OrderedDict
import itertools

from fuel.datasets import H5PYDataset
from picklable_itertools import iter_, chain
from fuel.datasets import Dataset
from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)
from fuel.transformers import Transformer

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
from machine_translation.model import BidirectionalEncoder, Decoder
from machine_translation.sampling import BleuValidator, Sampler, SamplingBase
from machine_translation.stream import (get_tr_stream, get_dev_stream,
                                        _ensure_special_tokens, MTSampleStreamTransformer,
                                        get_textfile_stream, _too_long, _length, PaddingWithEOS,
                                        _oov_to_unk)
from machine_translation.evaluation import sentence_level_bleu


from nnqe.dataset.preprocess import whitespace_tokenize


# In[ ]:

# create the graph which can sample from our model
# Note that we must sample instead of getting the 1-best or N-best, because we need the randomness to make the expected
# BLEU score make sense

exp_config = {
    'src_vocab_size': 20000,
    'trg_vocab_size': 20000,
    'enc_embed': 300,
    'dec_embed': 300,
    'enc_nhids': 800,
    'dec_nhids': 800,
    'saved_parameters': '/home/chris/projects/neural_mt/archived_models/BERTHA-TEST_Adam_wmt-multimodal_internal_data_dropout'+\
    '0.3_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/best_bleu_model_1455464992_BLEU31.61.npz',
    'src_vocab': '/home/chris/projects/neural_mt/archived_models/BERTHA-TEST_Adam_wmt-multimodal_internal_data_dropout0'+\
    '.3_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/vocab.en-de.en.pkl',
    'trg_vocab': '/home/chris/projects/neural_mt/archived_models/BERTHA-TEST_Adam_wmt-multimodal_internal_data_dropout0'+\
    '.3_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/vocab.en-de.de.pkl',
    'src_data': '/home/chris/projects/neural_mt/archived_models/BERTHA-TEST_Adam_wmt-multimodal_internal_data_dropout'+\
    '0.3_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/training_data/train.en.tok.shuf',
    'trg_data': '/home/chris/projects/neural_mt/archived_models/BERTHA-TEST_Adam_wmt-multimodal_internal_data_dropout'+\
    '0.3_ff_noiseFalse_search_model_en2es_vocab20000_emb300_rec800_batch15/training_data/train.de.tok.shuf',
    'unk_id':1,
    # Bleu script that will be used (moses multi-perl in this case)
    'bleu_script': '/home/chris/projects/neural_mt/test_data/sample_experiment/tiny_demo_dataset/multi-bleu.perl',
    # Optimization related ----------------------------------------------------
    # Batch size
    'batch_size': 5,
    # This many batches will be read ahead and sorted
    'sort_k_batches': 1,
    # Optimization step rule
    'step_rule': 'AdaDelta',
    # Gradient clipping threshold
    'step_clipping': 1.,
    # Std of weight initialization
    'weight_scale': 0.01,
    'seq_len': 40,
    'finish_after': 100
}


def get_sampling_model_and_input(exp_config):
    # Create Theano variables
    encoder = BidirectionalEncoder(
        exp_config['src_vocab_size'], exp_config['enc_embed'], exp_config['enc_nhids'])

    decoder = Decoder(
        exp_config['trg_vocab_size'], exp_config['dec_embed'], exp_config['dec_nhids'],
        exp_config['enc_nhids'] * 2,
        loss_function='min_risk'
    )

    # Create Theano variables
    logger.info('Creating theano variables')
    sampling_source_input = tensor.lmatrix('source')
    sampling_target_prefix_input = tensor.lmatrix('target')

    # Get beam search
    logger.info("Building sampling model")
    sampling_representation = encoder.apply(
        sampling_source_input, tensor.ones(sampling_source_input.shape))

    generated = decoder.generate(sampling_source_input, sampling_representation,
                                 target_prefix=sampling_target_prefix_input)

    # build the model that will let us get a theano function from the sampling graph
    logger.info("Creating Sampling Model...")
    sampling_model = Model(generated)

    # Set the parameters from a trained models
    logger.info("Loading parameters from model: {}".format(exp_config['saved_parameters']))
    # load the parameter values from an .npz file
    param_values = LoadNMT.load_parameter_values(exp_config['saved_parameters'], brick_delimiter='-')
    LoadNMT.set_model_parameters(sampling_model, param_values)

    return sampling_model, sampling_source_input, encoder, decoder

test_model, theano_sampling_input, train_encoder, train_decoder = get_sampling_model_and_input(exp_config)


# test that we can pull samples from the model
theano_sample_func = test_model.get_theano_function()

trg_vocab = cPickle.load(open(exp_config['trg_vocab']))
trg_vocab_size = exp_config['trg_vocab_size'] - 1
src_vocab = cPickle.load(open(exp_config['src_vocab']))
src_vocab_size = exp_config['src_vocab_size'] - 1

src_vocab = _ensure_special_tokens(src_vocab, bos_idx=0,
                                   eos_idx=src_vocab_size, unk_idx=exp_config['unk_id'])
trg_vocab = _ensure_special_tokens(trg_vocab, bos_idx=0,
                                   eos_idx=trg_vocab_size, unk_idx=exp_config['unk_id'])


# close over the sampling func and the trg_vocab to standardize the interface
# TODO: actually this should be a callable class with params (sampling_func, trg_vocab)
def sampling_func(source_seq, target_prefix, num_samples=1):

    def _get_true_length(seqs, vocab):
        try:
            lens = []
            for r in seqs.tolist():
                lens.append(r.index(vocab['</S>']) + 1)
            return lens
        except ValueError:
            return [seqs.shape[1] for _ in range(seqs.shape[0])]

    source_inputs = numpy.tile(source_seq[None, :], (num_samples, 1))
    target_prefix_inputs = numpy.tile(target_prefix[None, :], (num_samples, 1))
    # the output is [seq_len, batch]
    _1, outputs, _2, _3, costs = theano_sample_func(source_inputs, target_prefix_inputs)
    outputs = outputs.T

    # TODO: this step could be avoided by computing the samples mask in a different way
    lens = _get_true_length(outputs, trg_vocab)
    samples = [s[:l] for s,l in zip(outputs.tolist(), lens)]

    return samples


src_stream = get_textfile_stream(source_file=exp_config['src_data'], src_vocab=exp_config['src_vocab'],
                                         src_vocab_size=exp_config['src_vocab_size'])

# test_source_stream.sources = ('sources',)
trg_stream = get_textfile_stream(source_file=exp_config['trg_data'], src_vocab=exp_config['trg_vocab'],
                                         src_vocab_size=exp_config['trg_vocab_size'])

# Merge them to get a source, target pair
training_stream = Merge([src_stream,
                         trg_stream],
                         ('source', 'target'))


# Build a batched version of stream to read k batches ahead
training_stream = Batch(training_stream,
               iteration_scheme=ConstantScheme(
                   exp_config['batch_size']*exp_config['sort_k_batches']))

# Sort all samples in the read-ahead batch
training_stream = Mapping(training_stream, SortMapping(_length))

# Convert it into a stream again
training_stream = Unpack(training_stream)

# Construct batches from the stream with specified batch size
training_stream = Batch(
    training_stream, iteration_scheme=ConstantScheme(exp_config['batch_size']))

masked_stream = PaddingWithEOS(
    training_stream, [exp_config['src_vocab_size'] - 1, exp_config['trg_vocab_size'] - 1])

test_iter = masked_stream.get_epoch_iterator()

source, source_mask, target, target_mask = test_iter.next()


src_ivocab = {v:k for k,v in src_vocab.items()}
trg_ivocab = {v:k for k,v in trg_vocab.items()}

test_src = ['A', 'man', 'is', 'running', 'on', 'the', 'beach', '.']
test_src = numpy.array([src_vocab[w] for w in test_src])

test_prefix = ['Ein']
test_prefix = numpy.array([trg_vocab[w] for w in test_prefix])

SEN_INDEX=4
# import ipdb; ipdb.set_trace()
# TODO: HOW SHOULD PADDING BE HANDLED??
target_prefix = target[SEN_INDEX][:4]
s = sampling_func(source[SEN_INDEX], target_prefix, num_samples=10)
# s = sampling_func(test_src, test_prefix, num_samples=10)
# s = sampling_func(source[SEN_INDEX])


print('SOURCE: {}'.format(' '.join(src_ivocab[w] for w in source[SEN_INDEX])))
print('TARGET: {}'.format(' '.join(trg_ivocab[w] for w in target[SEN_INDEX])))
print('Target Prefix: {}'.format(' '.join(trg_ivocab[w] for w in target_prefix)))
# print('SOURCE: {}'.format(' '.join(src_ivocab[w] for w in test_src)))
# print('Target Prefix: {}'.format(' '.join(trg_ivocab[w] for w in test_prefix)))

for i in range(len(s)):
    print('TARGET {}: {}'.format(i, ' '.join(trg_ivocab[w] for w in s[i])))




