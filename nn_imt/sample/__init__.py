"""
Functions and classes for sampling from MMMT models

"""

from __future__ import print_function

import logging
import numpy
import operator
import os
import re
import signal
import time
import theano
import subprocess
import codecs
import tempfile

from blocks.extensions import SimpleExtension
from blocks.search import BeamSearch
from machine_translation.checkpoint import SaveLoadUtils

from nn_imt.evaluation import imt_f1_from_files

from subprocess import Popen, PIPE

logger = logging.getLogger(__name__)

# this is to let us use all of the sources in the fuel dev stream
# without needing to explicitly filter them
theano.config.on_unused_input = 'warn'

import numpy

class SampleFunc:

    def __init__(self, sample_func, vocab):
        self.sample_func = sample_func
        self.vocab = vocab

    # TODO: we may be able to make this function faster by passing multiple sources for sampling at the same damn time
    # TODO: or by avoiding the for loop somehow
    def __call__(self, source_seq, target_prefix, num_samples=1):

        source_inputs = numpy.tile(source_seq[None, :], (num_samples, 1))
        prefix_inputs = numpy.tile(target_prefix[None, :], (num_samples, 1))

        # the output is [seq_len, batch]
        # Note: we used model.get_theano_function to create the theano func
        # TODO: avoiding duplicates here is really messy, find a better way
        # TODO: one option is to filter the training stream to only contain training instances which have diverse samples
        num_tries = 10
        final_outputs = {}
        for _ in range(num_tries):
            if len(final_outputs) == num_samples:
                break
            _1, outputs, _2, _3, costs, word_probs = self.sample_func(source_inputs, prefix_inputs)
            word_probs = word_probs.transpose(1,0,2)
            for i,output in enumerate(outputs.T.tolist()):
                # convert word_probs to (batch, time, features)
                if tuple(output) not in final_outputs and len(final_outputs) != num_samples:
                    final_outputs[tuple(output)] = word_probs[i]

        outputs, word_probs = zip(*final_outputs.items())

        # if we didn't find enough unique samples, just sample again and use them all
        if len(outputs) < num_samples:
            _1, outputs, _2, _3, costs, word_probs = self.sample_func(source_inputs, prefix_inputs)
            # convert word_probs to (batch, time, features)
            outputs = outputs.T.tolist()
            word_probs = word_probs.transpose(1,0,2).tolist()

        try:
            assert len(outputs) == num_samples == len(word_probs), 'we must get the correct number of samples'
        except AssertionError:
            import ipdb;ipdb.set_trace()

        # get the probabilities of the words that were chosen
        sampled_word_probs = numpy.array([[word_probs[i][j][v] for j,v in enumerate(s)]
                                          for i, s in enumerate(outputs)])

        # TODO: this step could be avoided by computing the samples mask in a different way
        lens = self._get_true_length(outputs)
        samples = [s[:l] for s,l in zip(outputs, lens)]

        # Note: normalize by sequence-length? (probably not)
        true_word_probs = [c[:l] for c,l in zip(sampled_word_probs, lens)]
        log_seq_probs = [numpy.sum(numpy.log(s)) for s in true_word_probs]
        # Note: stay in log domain or not?? -- probably doesn't matter since we can't use this in a graph anyway
        seq_probs = numpy.array([numpy.exp(s) for s in log_seq_probs], dtype='float32')

        return samples, seq_probs

    def _get_true_length(self, seqs, seq_cutoff=1):
        try:
            lens = []
            if type(seqs) is not list:
                seqs = list(seqs)
            for r in seqs:
                lens.append(r.index(self.vocab['</S>']) + 1)
            return lens
        except ValueError:
            return [seq_cutoff for _ in range(len(seqs))]


class SamplingBase(object):
    """Utility class for BleuValidator and Sampler."""

    def _get_attr_rec(self, obj, attr):
        return self._get_attr_rec(getattr(obj, attr), attr) if hasattr(obj, attr) else obj

    def _get_true_length(self, seq, vocab):
        try:
            return seq.tolist().index(vocab['</S>']) + 1
        except ValueError:
            return len(seq)

    def _oov_to_unk(self, seq, vocab_size, unk_idx):
        return [x if x < vocab_size else unk_idx for x in seq]

    def _idx_to_word(self, seq, ivocab):
        return " ".join([ivocab.get(idx, "<UNK>") for idx in seq])

    def _initialize_dataset_info(self):
        # Get dictionaries, this may not be the practical way
        sources = self._get_attr_rec(self.main_loop, 'data_stream')

        # Load vocabularies and invert if necessary
        # WARNING: Source and target indices from data stream
        #  can be different
        #         if not hasattr(self, 'source_dataset'):
        #             self.source_dataset = sources.data_streams[0].dataset
        #         if not hasattr(self, 'target_dataset'):
        #             self.target_dataset = sources.data_streams[1].dataset
        if not hasattr(self, 'src_vocab'):
            self.src_vocab = self.source_dataset.dictionary
        if not hasattr(self, 'trg_vocab'):
            self.trg_vocab = self.target_dataset.dictionary
        if not hasattr(self, 'src_ivocab'):
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        if not hasattr(self, 'trg_ivocab'):
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
        if not hasattr(self, 'src_vocab_size'):
            self.src_vocab_size = len(self.src_vocab)


class Sampler(SimpleExtension, SamplingBase):
    """Random Sampling from model."""

    def __init__(self, model, data_stream, hook_samples=1,
                 src_vocab=None, trg_vocab=None, src_ivocab=None,
                 trg_ivocab=None, src_vocab_size=None, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.model = model
        self.hook_samples = hook_samples
        self.data_stream = data_stream
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_ivocab = src_ivocab
        self.trg_ivocab = trg_ivocab
        self.src_vocab_size = src_vocab_size
        self.is_synced = False

        self.sampling_fn = model.get_theano_function()

    def do(self, which_callback, *args):
        # Get dictionaries, this may not be the practical way
        sources = self._get_attr_rec(self.main_loop, 'data_stream')

        # Load vocabularies and invert if necessary
        # WARNING: Source and target indices from data stream
        #  can be different
        if not self.src_vocab:
            self.src_vocab = sources.data_streams[0].dataset.dictionary
        if not self.trg_vocab:
            self.trg_vocab = sources.data_streams[1].dataset.dictionary
        if not self.src_ivocab:
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
        if not self.src_vocab_size:
            self.src_vocab_size = len(self.src_vocab)

        # Randomly select source samples from the current batch
        # WARNING: Source and target indices from data stream
        #  can be different
        batch = args[0]
        batch_size = batch['source'].shape[0]
        hook_samples = min(batch_size, self.hook_samples)

        # TODO: this is problematic for boundary conditions, eg. last batch
        sample_idx = numpy.random.choice(
            batch_size, hook_samples, replace=False)

        # TODO: move to validation on suffixes, not the full target reference
        # TODO: print out the reference suffix and the prefix so we know what we are supposed to be generating
        src_batch = batch[self.main_loop.data_stream.mask_sources[0]]
        trg_batch = batch[self.main_loop.data_stream.mask_sources[1]]
        prefix_batch = batch[self.main_loop.data_stream.mask_sources[2]]
        suffix_batch = batch[self.main_loop.data_stream.mask_sources[3]]

        input_ = src_batch[sample_idx, :]
        target_ = trg_batch[sample_idx, :]
        prefix_ = prefix_batch[sample_idx, :]
        suffix_ = suffix_batch[sample_idx, :]

        # Sample
        print()
        for i in range(hook_samples):
            input_length = self._get_true_length(input_[i], self.src_vocab)

            prefix_length = self._get_true_length(prefix_[i], self.trg_vocab)
            target_length = self._get_true_length(target_[i], self.trg_vocab)

            inp = input_[i, :input_length]
            prefix = prefix_[i, :prefix_length]

            # outputs of self.sampling_fn:
            _1, outputs, _2, _3, costs = (self.sampling_fn(inp[None, :], prefix[None, :]))
            outputs = outputs.flatten()
            costs = costs.T

            sample_length = self._get_true_length(outputs, self.trg_vocab)

            print("Input : ", self._idx_to_word(input_[i][:input_length],
                                                self.src_ivocab))
            print("Prefix: ", self._idx_to_word(prefix_[i][:target_length],
                                                self.trg_ivocab))
            print("Suffix: ", self._idx_to_word(suffix_[i][:target_length],
                                                self.trg_ivocab))
            print("Sample: ", self._idx_to_word(outputs[:sample_length],
                                                self.trg_ivocab))
            print("Sample cost: ", costs[:sample_length].sum())
            print()


# Note: this validator depends upon us having a file of references, but for IMT the references are generated on the fly
# Note: therefore we need to write all the suffixes to a temporary file and validate against that
# WORKING: we actually want to evaluate against whichever samples are currently on the stream
class BleuValidator(SimpleExtension, SamplingBase):
    """Implements early stopping based on BLEU score."""

    def __init__(self, source_sentence, target_prefix, samples, model, data_stream,
                 config, src_vocab=None, trg_vocab=None, n_best=1, track_n_models=1,
                 normalize=True, **kwargs):
        super(BleuValidator, self).__init__(**kwargs)
        self.source_sentence = source_sentence
        self.target_prefix = target_prefix

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.samples = samples
        self.model = model
        self.data_stream = data_stream
        self.config = config
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.normalize = normalize
        self.verbose = config.get('val_set_out', None)

        # Helpers
        self.best_models = []
        self.val_bleu_curve = []
        self.beam_search = BeamSearch(samples=samples)
        self.multibleu_cmd = ['perl', self.config['bleu_script'],
                              self.config['val_set_grndtruth'], '<']

        # Create save directory if it does not exist
        if not os.path.exists(self.config['saveto']):
            os.makedirs(self.config['saveto'])

        if self.config['reload']:
            try:
                bleu_score = numpy.load(os.path.join(self.config['saveto'],
                                                     'val_bleu_scores.npz'))
                self.val_bleu_curve = bleu_score['bleu_scores'].tolist()

                # Track n best previous bleu scores
                for i, bleu in enumerate(
                        sorted(self.val_bleu_curve, reverse=True)):
                    if i < self.track_n_models:
                        self.best_models.append(ModelInfo(bleu, key='BLEU'))
                logger.info("BleuScores Reloaded")
            except:
                logger.info("BleuScores not Found")

    def do(self, which_callback, *args):

        # Track validation burn in
        if self.main_loop.status['iterations_done'] <= self.config['val_burn_in']:
            return

        # Evaluate the model
        bleu_score = self._evaluate_model()
        # add an entry to the log
        self.main_loop.log.current_row['validation_set_bleu_score'] = bleu_score
        # save if necessary
        self._save_model(bleu_score)

    def _evaluate_model(self):
        # Set in the superclass -- SamplingBase
        if not hasattr(self, 'target_dataset'):
            self._initialize_dataset_info()

        #         self.unk_sym = self.target_dataset.unk_token
        #         self.eos_sym = self.target_dataset.eos_token

        self.unk_sym = '<UNK>'
        self.eos_sym = '</S>'
        self.unk_idx = self.trg_vocab[self.unk_sym]
        self.eos_idx = self.trg_vocab[self.eos_sym]

        logger.info("Started Validation: ")
        val_start_time = time.time()
        mb_subprocess = Popen(self.multibleu_cmd, stdin=PIPE, stdout=PIPE)
        total_cost = 0.0

        if self.verbose:
            ftrans = open(self.config['val_set_out'], 'w')

        print('LENGTH OF DEV STREAM: {}'.format(len(list(self.data_stream.get_epoch_iterator()))))
        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            """
            Load the sentence, retrieve the sample, write to file
            """

            # Note that the indices of source and target in the datastream are hard-coded
            # currently our datastream is (source,target,prefix,suffix)
            seq = self._oov_to_unk(
                line[0], self.config['src_vocab_size'], self.unk_idx)

            target_prefix = line[2]

            input_ = numpy.tile(seq, (self.config['beam_size'], 1))
            prefix_input_ = numpy.tile(target_prefix, (self.config['beam_size'], 1))

            # draw sample, checking to ensure we don't get an empty string back
            # beam search param names come from WHERE??
            trans, costs = self.beam_search.search(
                input_values={self.source_sentence: input_,
                              self.target_prefix: prefix_input_},
                max_length=3*len(seq), eol_symbol=self.eos_idx,
                ignore_first_eol=False)

            # normalize costs according to the sequence lengths
            if self.normalize:
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths

            nbest_idx = numpy.argsort(costs)[:self.n_best]

            for j, best in enumerate(nbest_idx):
                try:
                    total_cost += costs[best]
                    trans_out = trans[best]

                    # convert idx to words
		    #print('input_seq: {}'.format(seq))
		    #print('input_prefix: {}'.format(target_prefix))
		    #print('trans_out_raw: {}'.format(trans_out))
                    trans_out = self._idx_to_word(trans_out, self.trg_ivocab)
		    #print('trans_out_text: {}'.format(trans_out))

                except ValueError:
                    logger.info(
                        "Can NOT find a translation for line: {}".format(i+1))
                    trans_out = '<UNK>'

                if j == 0:
                    # Write to subprocess and file if it exists
                    print(trans_out, file=mb_subprocess.stdin)
                    if self.verbose:
                        print(trans_out, file=ftrans)

                if i != 0 and i % 100 == 0:
                    logger.info(
                        "Translated {} lines of validation set...".format(i))

                mb_subprocess.stdin.flush()

        logger.info("Total cost of the validation: {}".format(total_cost))
        self.data_stream.reset()
        if self.verbose:
            ftrans.close()

        # send end of file, read output.
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        logger.info(stdout)
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        logger.info("Validation Took: {} minutes".format(
            float(time.time() - val_start_time) / 60.))
        assert out_parse is not None

        # extract the score
        bleu_score = float(out_parse.group()[6:])
        self.val_bleu_curve.append(bleu_score)
        logger.info(bleu_score)
        mb_subprocess.terminate()

        return bleu_score

    def _is_valid_to_save(self, bleu_score):
        if not self.best_models or min(self.best_models,
                                       key=operator.attrgetter('score')).score < bleu_score:
            return True
        return False

    def _save_model(self, bleu_score):
        if self._is_valid_to_save(bleu_score):
            model = ModelInfo(bleu_score, self.config['saveto'], key='BLEU')

            # Manage n-best model list first
            if len(self.best_models) >= self.track_n_models:
                old_model = self.best_models[0]
                if old_model.path and os.path.isfile(old_model.path):
                    logger.info("Deleting old model %s" % old_model.path)
                    os.remove(old_model.path)
                self.best_models.remove(old_model)

            self.best_models.append(model)
            self.best_models.sort(key=operator.attrgetter('score'))

            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            logger.info("Saving new model {}".format(model.path))

            SaveLoadUtils.save_parameter_values(self.main_loop.model.get_parameter_values(), model.path)
            numpy.savez(
                os.path.join(self.config['saveto'], 'val_bleu_scores.npz'),
                bleu_scores=self.val_bleu_curve)
            signal.signal(signal.SIGINT, s)


# TODO: implement Meteor for IMT validation
class MeteorValidator(SimpleExtension, SamplingBase):
    """Implements early stopping based on METEOR score."""

    def __init__(self, source_sentence, initial_state_context, samples, model, data_stream,
                 config, src_vocab=None, trg_vocab=None, n_best=1, track_n_models=1,
                 normalize=True, **kwargs):
        super(MeteorValidator, self).__init__(**kwargs)
        self.source_sentence = source_sentence
        self.initial_context = initial_state_context

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.samples = samples
        self.model = model
        self.data_stream = data_stream
        self.config = config
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.normalize = normalize
        self.verbose = config.get('val_set_out', None)

        # Helpers
        self.best_models = []
        self.val_meteor_curve = []
        self.beam_search = BeamSearch(samples=samples)

        # Info for Meteor
        self.target_language = self.config['target_lang']
        self.meteor_directory = self.config['meteor_directory']

        # Create save directory if it does not exist
        if not os.path.exists(self.config['saveto']):
            os.makedirs(self.config['saveto'])

        if self.config['reload']:
            try:
                meteor_score = numpy.load(os.path.join(self.config['saveto'],
                                                       'val_meteor_scores.npz'))
                self.val_meteor_curve = meteor_score['meteor_scores'].tolist()

                # Track n best previous meteor scores
                for i, meteor in enumerate(
                        sorted(self.val_meteor_curve, reverse=True)):
                    if i < self.track_n_models:
                        self.best_models.append(ModelInfo(meteor, key='METEOR'))
                logger.info("MeteorScores Reloaded")
            except:
                logger.info("MeteorScores not Found")

    def do(self, which_callback, *args):

        # Track validation burn in
        if self.main_loop.status['iterations_done'] <= self.config['val_burn_in']:
            return

        # Evaluate the model
        meteor_score = self._evaluate_model()
        # add an entry to the log
        self.main_loop.log.current_row['validation_set_meteor_score'] = meteor_score
        # save if necessary
        self._save_model(meteor_score)

    # TODO: if we are evaluating both BLEU and METEOR, we shouldn't need to translate twice!!
    def _evaluate_model(self):
        # Set in the superclass -- SamplingBase
        if not hasattr(self, 'target_dataset'):
            self._initialize_dataset_info()

        self.unk_sym = '<UNK>'
        self.eos_sym = '</S>'
        self.unk_idx = self.trg_vocab[self.unk_sym]
        self.eos_idx = self.trg_vocab[self.eos_sym]

        logger.info("Started Validation: ")
        val_start_time = time.time()

        ref_file = self.config['val_set_grndtruth']
        # TODO: write all hyps to temp file for meteor
        trg_hyp_file = tempfile.NamedTemporaryFile(delete=False)

        if self.verbose:
            ftrans = codecs.open(self.config['val_set_out'], 'w', encoding='utf8')

        total_cost = 0.0
        with codecs.open(trg_hyp_file.name, 'w', encoding='utf8') as hyps_out:
            for i, line in enumerate(self.data_stream.get_epoch_iterator()):
                """
                Load the sentence, retrieve the sample, write to file
                """

                # TODO: the section with beam search and translation is shared by all validators
                seq = self._oov_to_unk(
                    line[0], self.config['src_vocab_size'], self.unk_idx)

                # TODO: the index is different when we are doing min-risk
                initial_state_context = line[-1]

                input_ = numpy.tile(seq, (self.config['beam_size'], 1))
                context_input_ = numpy.tile(initial_state_context, (self.config['beam_size'], 1))

                # draw sample, checking to ensure we don't get an empty string back
                trans, costs = self.beam_search.search(
                    input_values={self.source_sentence: input_,
                                  self.initial_context: context_input_},
                        max_length=3*len(seq), eol_symbol=self.eos_idx,
                        ignore_first_eol=False)

                # normalize costs according to the sequence lengths
                if self.normalize:
                    lengths = numpy.array([len(s) for s in trans])
                    costs = costs / lengths

                nbest_idx = numpy.argsort(costs)[:self.n_best]
                for j, best in enumerate(nbest_idx):
                    try:
                        total_cost += costs[best]
                        trans_out = trans[best]

                        # convert idx to words
                        trans_out = self._idx_to_word(trans_out, self.trg_ivocab)

                    except ValueError:
                        logger.info(
                            "Can NOT find a translation for line: {}".format(i+1))
                        trans_out = '<UNK>'

                    if j == 0:
                        # Write to subprocess and file if it exists
                        hyps_out.write(trans_out.decode('utf8') + '\n')
                        if self.verbose:
                            print(trans_out.decode('utf8'), file=ftrans)

                if i != 0 and i % 100 == 0:
                    logger.info(
                        "Translated {} lines of validation set...".format(i))

            logger.info("Total cost of the validation: {}".format(total_cost))

            self.data_stream.reset()
            if self.verbose:
                ftrans.close()

        meteor_cmd = ['java', '-Xmx4G', '-jar', os.path.join(self.meteor_directory, 'meteor-1.5.jar'),
                      trg_hyp_file.name, ref_file, '-l', self.target_language, '-norm']

        meteor_output = subprocess.check_output(meteor_cmd)
        meteor_score = float(meteor_output.strip().split('\n')[-1].split()[-1])
        logger.info('METEOR SCORE: {}'.format(meteor_score))

        logger.info("Meteor Validation Took: {} minutes".format(float(time.time() - val_start_time) / 60.))

        return meteor_score

    def _is_valid_to_save(self, meteor_score):
        if not self.best_models or min(self.best_models,
                                       key=operator.attrgetter('score')).score < meteor_score:
            return True
        return False

    def _save_model(self, meteor_score):
        if self._is_valid_to_save(meteor_score):
            model = ModelInfo(meteor_score, self.config['saveto'], key='METEOR')

            # Manage n-best model list first
            if len(self.best_models) >= self.track_n_models:
                old_model = self.best_models[0]
                if old_model.path and os.path.isfile(old_model.path):
                    logger.info("Deleting old model %s" % old_model.path)
                    os.remove(old_model.path)
                self.best_models.remove(old_model)

            self.best_models.append(model)
            self.best_models.sort(key=operator.attrgetter('score'))

            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            logger.info("Saving new model {}".format(model.path))

            SaveLoadUtils.save_parameter_values(self.main_loop.model.get_parameter_values(), model.path)
            numpy.savez(
                os.path.join(self.config['saveto'], 'val_meteor_scores.npz'),
                meteor_scores=self.val_meteor_curve)
            signal.signal(signal.SIGINT, s)

class IMT_F1_Validator(SimpleExtension, SamplingBase):
    """Implements early stopping based on METEOR score."""

    def __init__(self, source_sentence, target_prefix, samples, model, data_stream,
                 config, src_vocab=None, trg_vocab=None, n_best=1, track_n_models=1,
                 normalize=True, **kwargs):
        super(IMT_F1_Validator, self).__init__(**kwargs)
        self.source_sentence = source_sentence
        self.target_prefix = target_prefix

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.samples = samples
        self.model = model
        self.data_stream = data_stream
        self.config = config
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.normalize = normalize
        self.verbose = config.get('val_set_out', None)

        # Helpers
        self.best_models = []
        self.val_imt_f1_curve = []
        self.beam_search = BeamSearch(samples=samples)

        # Info for Meteor
        self.target_language = self.config['target_lang']

        # Create save directory if it does not exist
        if not os.path.exists(self.config['saveto']):
            os.makedirs(self.config['saveto'])

        if self.config['reload']:
            try:
                imt_f1_score = numpy.load(os.path.join(self.config['saveto'],
                                                       'val_imt_f1_scores.npz'))
                self.val_imt_f1_curve = imt_f1_score['imt_f1_scores'].tolist()

                # Track n best previous f1_bad scores
                for i, imt_f1_val in enumerate(
                        sorted(self.val_imt_f1_curve, reverse=True)):
                    if i < self.track_n_models:
                        self.best_models.append(ModelInfo(imt_f1_val, key='IMT_F1'))
                logger.info("IMT_F1_Scores Reloaded")
            except:
                logger.info("IMT_F1_Scores not found")

    def do(self, which_callback, *args):

        # Track validation burn in
        if self.main_loop.status['iterations_done'] <= self.config['val_burn_in']:
            return

        # Evaluate the model
        imt_f1_score = self._evaluate_model()
        # add an entry to the log
        self.main_loop.log.current_row['validation_set_imt_f1_score'] = imt_f1_score
        # save if necessary
        self._save_model(imt_f1_score)

    # TODO: if we are evaluating both BLEU and METEOR, we shouldn't need to translate twice!!
    def _evaluate_model(self):
        # Set in the superclass -- SamplingBase
        if not hasattr(self, 'target_dataset'):
            self._initialize_dataset_info()

        self.unk_sym = '<UNK>'
        self.eos_sym = '</S>'
        self.unk_idx = self.trg_vocab[self.unk_sym]
        self.eos_idx = self.trg_vocab[self.eos_sym]

        logger.info("Started Validation: ")
        val_start_time = time.time()

        ref_file = self.config['val_set_grndtruth']

        trg_hyp_file = tempfile.NamedTemporaryFile(delete=False)

        if self.verbose:
            ftrans = codecs.open(self.config['val_set_out'], 'w', encoding='utf8')

        total_cost = 0.0
        with codecs.open(trg_hyp_file.name, 'w', encoding='utf8') as hyps_out:
            for i, line in enumerate(self.data_stream.get_epoch_iterator()):
                """
                Load the sentence, retrieve the sample, write to file
                """

                # TODO: the section with beam search and translation is shared by all validators
                # WORKING: switch this to IMT prefix validation
                # Note that the indices of source and target in the datastream are hard-coded
                # currently our datastream is (source,target,prefix,suffix)
                seq = self._oov_to_unk(
                    line[0], self.config['src_vocab_size'], self.unk_idx)

                target_prefix = line[2]

                input_ = numpy.tile(seq, (self.config['beam_size'], 1))
                prefix_input_ = numpy.tile(target_prefix, (self.config['beam_size'], 1))

                # draw sample, checking to ensure we don't get an empty string back
                # beam search param names come from WHERE??
                trans, costs = self.beam_search.search(
                    input_values={self.source_sentence: input_,
                                  self.target_prefix: prefix_input_},
                    max_length=3*len(seq), eol_symbol=self.eos_idx,
                    ignore_first_eol=False)

                # normalize costs according to the sequence lengths
                if self.normalize:
                    lengths = numpy.array([len(s) for s in trans])
                    costs = costs / lengths

                nbest_idx = numpy.argsort(costs)[:self.n_best]
                for j, best in enumerate(nbest_idx):
                    try:
                        total_cost += costs[best]
                        trans_out = trans[best]

                        # convert idx to words
                        trans_out = self._idx_to_word(trans_out, self.trg_ivocab)

                    except ValueError:
                        logger.info(
                            "Can NOT find a translation for line: {}".format(i+1))
                        trans_out = '<UNK>'

                    if j == 0:
                        # Write to subprocess and file if it exists
                        hyps_out.write(trans_out.decode('utf8') + '\n')
                        if self.verbose:
                            print(trans_out.decode('utf8'), file=ftrans)

                if i != 0 and i % 100 == 0:
                    logger.info(
                        "Translated {} lines of validation set...".format(i))

            logger.info("Total cost of the validation: {}".format(total_cost))

            self.data_stream.reset()
            if self.verbose:
                ftrans.close()

        imt_f1_score, imt_precision, imt_recall = imt_f1_from_files(trg_hyp_file.name, ref_file)

        logger.info("IMT F1 Validation Took: {} minutes".format(float(time.time() - val_start_time) / 60.))
        logger.info("IMT F1: {}, Precision: {}, Recall: {}".format(imt_f1_score, imt_precision, imt_recall))

        return imt_f1_score

    def _is_valid_to_save(self, imt_f1_score):
        if not self.best_models or min(self.best_models,
                                       key=operator.attrgetter('score')).score < imt_f1_score:
            return True
        return False

    def _save_model(self, imt_f1_score):
        if self._is_valid_to_save(imt_f1_score):
            model = ModelInfo(imt_f1_score, self.config['saveto'], key='IMT_F1')

            # Manage n-best model list first
            if len(self.best_models) >= self.track_n_models:
                old_model = self.best_models[0]
                if old_model.path and os.path.isfile(old_model.path):
                    logger.info("Deleting old model %s" % old_model.path)
                    os.remove(old_model.path)
                self.best_models.remove(old_model)

            self.best_models.append(model)
            self.best_models.sort(key=operator.attrgetter('score'))

            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            logger.info("Saving new model {}".format(model.path))

            SaveLoadUtils.save_parameter_values(self.main_loop.model.get_parameter_values(), model.path)
            numpy.savez(
                os.path.join(self.config['saveto'], 'val_imt_f1_scores.npz'),
                imt_f1_scores=self.val_imt_f1_curve)
            signal.signal(signal.SIGINT, s)


class ModelInfo:
    """Utility class to keep track of evaluated models."""

    def __init__(self, score, path=None, key='SCORE'):
        self.score = score
        self.key = key
        self.path = self._generate_path(path)

    def _generate_path(self, path):
        gen_path = os.path.join(
            path, 'best_model_%d_%s%.2f.npz' %
                  (int(time.time()), self.key, self.score) if path else None)
        return gen_path


