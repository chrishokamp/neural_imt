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
        logger.info('Acquired lock')
        lock.acquire()

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

        print "Lock release"
        lock.release()

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

    return jsonify({'ranked_completions': translations})


def prefix_decode(source_lang, target_lang, source_sentence, target_prefix, n_best=5):
    '''
    Call the decoder, get a suffix hypothesis
    :param source_sentence:
    :param target_prefix:
    :return: translations
    '''
    # TODO: move tokenization logic to server??
    # TODO: persistent tokenizer -- don't spin a new process each time
    predictor = app.predictors[(source_lang, target_lang)]

    best_n_hyps, best_n_costs, best_n_glimpses, best_n_word_level_costs, best_n_confidences, src_in = predictor.predict_segment(source_sentence, target_prefix=target_prefix,
                                                        tokenize=True, detokenize=True, n_best=n_best, max_length=predictor.max_length)

    # TODO: we _must_ add subword configuration as well -- we need to apply subword, then re-concat afterwards
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
    import ipdb; ipdb.set_trace()
    # TODO: a different kind of config for servers -- maps language pairs to NMT configuration files
    # TODO: server instantiates a predictor for each config, and hashes them by language pair tuples -- i.e. (en,fr)
    # TODO: make the indexing API visible via the predictor
    # TODO: NMT passes in a dict of predictors, keys are tuples (source_lang, target_lang)
    app.predictors = predictors

    logger.info('Server starting on port: {}'.format(port))
    logger.info('navigate to: http://localhost:{}/neural_MT_demo to see the system demo'.format(port))
    app.run(debug=True, port=port, host='127.0.0.1', threaded=True)

