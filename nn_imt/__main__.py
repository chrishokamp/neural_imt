from __future__ import print_function
import argparse
import logging
import pprint
import codecs
import re
import os
import time
from subprocess import Popen, PIPE

from machine_translation import configurations

from nn_imt import main, IMTPredictor, split_refs_into_prefix_suffix_files

from nn_imt.stream import get_tr_stream_with_prefixes, get_dev_stream_with_prefixes
from nn_imt.sample import SamplingBase
from nn_imt.evaluation import imt_f1_from_files

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument("exp_config",
                    help="Path to the yaml config file for your experiment")
parser.add_argument("-m", "--mode", default='train',
                    help="The mode we are in [train,predict,server] -- default=train")
parser.add_argument("--bokeh",  default=False, action="store_true",
                    help="Use bokeh server for plotting")

if __name__ == "__main__":
    # Get configurations for model

    # WORKING: set up IMT evaluation and BLEU early stopping
    # training examples are triples of (source, prefix, completion)
    # so references are now completions, not the complete reference
    # THINKING: how to account for completions when user is inside a word? -- character NMT on target-side is the most satisfying,
    # but is difficult to implement

    args = parser.parse_args()
    arg_dict = vars(args)
    configuration_file = arg_dict['exp_config']
    mode = arg_dict['mode']
    logger.info('Running Neural Machine Translation in mode: {}'.format(mode))
    config_obj = configurations.get_config(configuration_file)
    # add the config file name into config_obj
    config_obj['config_file'] = configuration_file
    logger.info("Model Configuration:\n{}".format(pprint.pformat(config_obj)))

    if mode == 'train':
        # Get data streams and call main
        training_stream, src_vocab, trg_vocab = get_tr_stream_with_prefixes(**config_obj)
        dev_stream = get_dev_stream_with_prefixes(**config_obj)

        trg_ivocab = {v: k for k, v in trg_vocab.items()}
        if not os.path.isdir(config_obj['model_save_directory']):
            os.mkdir(config_obj['model_save_directory'])

        suffix_ref_filename = os.path.join(config_obj['model_save_directory'], 'reference_suffixes.out')

        sampling_base = SamplingBase()
        with codecs.open(suffix_ref_filename, 'w') as suffix_refs:
            for l in list(dev_stream.get_epoch_iterator()):
                # currently our datastream is (source,target,prefix,suffix)
                suffix = l[-1]
                suffix_text = sampling_base._idx_to_word(suffix, trg_ivocab)
                assert len(suffix_text) > 0, 'reference cannot be empty'
                suffix_refs.write(suffix_text + '\n')

        config_obj['val_set_grndtruth'] = suffix_ref_filename

        main(config_obj, training_stream, dev_stream, src_vocab, trg_vocab, args.bokeh)

    elif mode == 'predict':
        predictor = IMTPredictor(config_obj)
        # Two different cases for prediction
        # (1) user provides two files: sources, and prefixes
        # (2) user provides sources and targets -- the targets are split into prefixes and the sources are duplicated
        # in the same way as they are for training

        # the case where user provided a file of prefixes
        prediction_prefixes = config_obj.get('test_prefixes', None)

        # TODO: move this logic outside so we can reuse it in evaluation
        if not prediction_prefixes:
            try:
                prediction_refs = config_obj['test_gold_refs']
            except KeyError:
                print('If you do not provide a prefix file, you must provide a file of complete references')
                raise

            sources_file, prediction_prefixes, _ = split_refs_into_prefix_suffix_files(prediction_refs, config_obj)
        else:
            sources_file = config_obj['test_set']

        predictor.predict_files(sources_file, prediction_prefixes, output_file=config_obj['translated_output_file'])
        logger.info('Done Predicting')

    elif mode == 'evaluate':
        logger.info("Started Evaluation: ")
        val_start_time = time.time()

        # TODO: support evaluation for IMT
        # WORKING: IMT evaluation is slightly more involved because we may need to expand the dev set into
        # WORKING: (source, prefix, suffix)
        # WORKING: in practice this also means that validation takes much longer, so we should probably start with a
        # WORKING: smaller dev set, which is a sample of the full dev set

        # create the control function which will run evaluation
        # currently available evaluation metrics: 'bleu', 'meteor', 'imt_f1', 'imt_ndcg'
        evaluation_metrics = config_obj.get('evaluation_metrics', ['bleu'])

        # translate if necessary, write output file, call external evaluation tools and show output
        translated_output_file = config_obj.get('translated_output_file', None)
        if translated_output_file is not None and os.path.isfile(translated_output_file):
            logger.info('{} already exists, so I\'m evaluating the BLEU score of this file with respect to the ' +
                        'reference that you provided: {}'.format(translated_output_file,
                        config_obj['test_gold_refs']))
            references_file = config_obj['test_gold_refs']
        else:
            predictor = IMTPredictor(config_obj)

            # the case where user provided a file of prefixes
            prediction_prefixes = config_obj.get('test_prefixes', None)

            # TODO: move this logic outside so we can reuse it in evaluation
            if not prediction_prefixes:
                try:
                    prediction_refs = config_obj['test_gold_refs']
                except KeyError:
                    print('If you do not provide a prefix file, you must provide a file of complete references')
                    raise

                sources_file, prediction_prefixes, references_file = split_refs_into_prefix_suffix_files(prediction_refs, config_obj)
            else:
                sources_file = config_obj['test_set']
                references_file = config_obj['test_gold_refs']

            predictor.predict_files(sources_file, prediction_prefixes, output_file=config_obj['translated_output_file'])
            logger.info('Done translating, now I will evaluate the metrics: {}'.format(evaluation_metrics))

        logger.info("Started Evaluation: ")
        val_start_time = time.time()

        if 'bleu' in evaluation_metrics:
            # TODO: add a sanity check that hyps and refs have the same number of lines, and no refs or hyps are empty
            translated_output_file = config_obj.get('translated_output_file', None)
            # get gold refs
            multibleu_cmd = ['perl', config_obj['bleu_script'],
                             references_file, '<']
            mb_subprocess = Popen(multibleu_cmd, stdin=PIPE, stdout=PIPE)
            with codecs.open(translated_output_file, encoding='utf8') as hyps:
                for l in hyps.read().strip().split('\n'):
                    # send the line to the BLEU script
                    print(l.encode('utf8'), file=mb_subprocess.stdin)
            mb_subprocess.stdin.flush()

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
            logger.info('BLEU SCORE: {}'.format(bleu_score))
            mb_subprocess.terminate()
        if 'imt_f1' in evaluation_metrics:
            translated_output_file = config_obj.get('translated_output_file', None)
            imt_f1_score, precision, recall = imt_f1_from_files(translated_output_file, references_file)
            logger.info('IMT F1 SCORE: {}, precision: {}, recall: {}'.format(imt_f1_score, precision, recall))

    elif mode == 'server':

        import sys
        sys.path.append('.')
        from server import run_nmt_server

        # start restful server and log its port
        predictor = IMTPredictor(config_obj)

        # TODO: change to run_imt_server
        run_nmt_server(predictor)


