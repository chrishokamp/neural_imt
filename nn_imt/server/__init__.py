import logging
import os
import re
import threading
import json

from flask import Flask, request, render_template, jsonify, abort
from wtforms import Form, TextAreaField, validators

logger = logging.getLogger(__name__)

app = Flask(__name__)
# this needs to be set before we actually run the server
app.predictors = None

path_to_this_dir = os.path.dirname(os.path.abspath(__file__))
app.template_folder = os.path.join(path_to_this_dir, 'templates')

lock = threading.Lock()

class NeuralMTDemoForm(Form):
    sentence = TextAreaField('Write the sentence here:',
                             [validators.Length(min=1, max=100000)])
    prefix = TextAreaField('Write the target prefix here:',
                             [validators.Length(min=1, max=100)])
    target_text = ''


# TODO: multiple instances of the same model, delegate via thread queue?
# TODO: supplementary info via endpoint -- softmax confidence, cumulative confidence, etc...
# TODO: online updating via cache
# TODO: require source and target language specification
@app.route('/neural_MT_demo', methods=['GET', 'POST'])
def neural_mt_demo():
    form = NeuralMTDemoForm(request.form)
    if request.method == 'POST' and form.validate():

        # TODO: delegate to prefix decoder function here
        source_text = form.sentence.data # Problems in Spanish with 'A~nos. E'
        target_text = form.prefix.data

        # TODO: is locking actually a good idea?
        # logger.info('Acquired lock')
        # lock.acquire()

        source_sentence = source_text.encode('utf-8')
        target_prefix = target_text.encode('utf-8')

        # translations, costs = app.predictor.predict_segment(source_sentence, tokenize=True, detokenize=True)
        translations, costs = app.predictor.predict_segment(source_sentence, target_prefix=target_prefix,
                                                            tokenize=True, detokenize=True, n_best=10)

        output_text = ''
        for hyp in translations:
            output_text += hyp + '\n'

        form.target_text = output_text.decode('utf-8')
        logger.info('detokenized translations:\n {}'.format(output_text))

        # TODO: is locking actually a good idea?
        # print "Lock release"
        # lock.release()

    return render_template('neural_MT_demo.html', form=form)


@app.route('/nimt', methods=['GET', 'POST'])
def neural_mt_prefix_decoder():
    # TODO: parse request object, remove form
    if request.method == 'POST':
        request_data = request.get_json()
        print(request_data)
        source_lang = request_data['source_lang']
        target_lang = request_data['target_lang']

        # WORKING: languages from express endpoint
        # WORKING: return 404 if language pair isn't available
        if (source_lang, target_lang) not in app.predictors:
            logger.error('IMT server does not have a model for: {}'.format((source_lang, target_lang)))
            abort(404)

        source_sentence = request_data['source_sentence'] # Problems in Spanish with 'A~nos. E'
        target_prefix = request_data['target_prefix'] # Problems in Spanish with 'A~nos. E'

        logger.info('Acquired lock')
        lock.acquire()

        translations = prefix_decode(source_lang, target_lang, source_sentence, target_prefix)

        # output_text = ''
        # for hyp in translations:
        #     output_text += hyp + '\n'

        # form.target_text = output_text.decode('utf-8')
        # logger.info('detokenized translations:\n {}'.format(output_text))

        print "Lock release"
        lock.release()
        request_time = request_data.get('request_time', 0)

    return jsonify({'ranked_completions': translations, 'request_time': request_time})


def prefix_decode(source_lang, target_lang, source_sentence, target_prefix, n_best=5):
    '''
    Call the decoder, get a suffix hypothesis
    :param source_sentence:
    :param target_prefix:
    :return: translations
    '''
    predictor = app.predictors[(source_lang, target_lang)]
    subword = False
    if predictor.BPE is not None:
        subword = True

    best_n_hyps, best_n_costs, best_n_glimpses, best_n_word_level_costs, best_n_confidences, src_in = predictor.predict_segment(source_sentence, target_prefix=target_prefix,
                                                        tokenize=True, detokenize=True, n_best=n_best, max_length=predictor.max_length, subword_encode=subword)

    # TODO: we _must_ add subword configuration as well -- we need to apply subword, then re-concat afterwards
    # TODO: subword is optional -- push this logic to predictor
    # remove EOS and normalize subword
    def _postprocess(hyp):
        hyp = re.sub("</S>$", "", hyp)
        # Note the order of the next two lines is important
        hyp = re.sub("\@\@ ", "", hyp)
        hyp = re.sub("\@\@", "", hyp)
        return hyp

    postprocessed_hyps = [_postprocess(h) for h in best_n_hyps]

    return postprocessed_hyps


def run_imt_server(predictors, port=5000):
    # TODO: persistent subword option in predictors
    app.predictors = predictors

    logger.info('Server starting on port: {}'.format(port))
    logger.info('navigate to: http://localhost:{}/neural_MT_demo to see the system demo'.format(port))
    app.run(debug=True, port=port, host='127.0.0.1', threaded=True)
    # app.run(debug=True, port=port, host='127.0.0.1', threaded=False)

