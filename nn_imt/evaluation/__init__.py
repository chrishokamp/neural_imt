import codecs
import numpy


# sentence level imt_f1 scoring
# TODO: we really shouldn't need to pass the source here, this is because of the mteval_v13 interface
def sentence_level_imt_f1(src, ref, samples, **kwargs):

    scores, _, _ = zip(*[imt_f1(sample, ref) for sample in samples])
    one_minus_scores = [1.-s for s in scores]

    return one_minus_scores

def num_prd(hyp, ref):
    """
    compute the number of correctly predicted words in the suffix

    :return: the length (in words) of the correct section of the suffix

    """

    # if both are empty you didn't predict anything correct
    if len(hyp) == 0 or len(ref) == 0:
        return 0.

    match_len = 0.
    for h_sym, r_sym in zip(hyp, ref):
        if h_sym == r_sym:
            match_len += 1.
        else:
            break

    return match_len

def num_prd_from_files(hyp_file, ref_file):
    """
    Compute average WPA over the provided files
    :param hyp_file:
    :param ref_file:
    :return:

    Note: input files are assumed to be whitespace-tokenized, if this isn't the case this metric will be wrong

    """

    with codecs.open(hyp_file, encoding='utf8') as hyp_f:
        hyps = hyp_f.read().strip().split('\n')
        hyps = [l.split() for l in hyps]

        with codecs.open(ref_file, encoding='utf8') as ref_f:
            refs = ref_f.read().strip().split('\n')
            refs = [l.split() for l in refs]

    scores = []
    for hyp, ref in zip(hyps, refs):
        scores.append(num_prd(hyp, ref))

    return numpy.mean(scores)


def wpa(hyp, ref):
    """
    compute first word prediction accuracy

    :return: 1. if first word is correct, 0. otherwise

    """

    # if both are empty, this is a perfect match
    if len(hyp) == 0 and len(ref) == 0:
        return 1.

    if hyp[0] == ref[0]:
        return 1.
    else:
        return 0.


def wpa_from_files(hyp_file, ref_file):
    """
    Compute average WPA over the provided files
    :param hyp_file:
    :param ref_file:
    :return:

    Note: input files are assumed to be whitespace-tokenized, if this isn't the case this metric will be wrong

    """

    with codecs.open(hyp_file, encoding='utf8') as hyp_f:
        hyps = hyp_f.read().strip().split('\n')
        hyps = [l.split() for l in hyps]

        with codecs.open(ref_file, encoding='utf8') as ref_f:
            refs = ref_f.read().strip().split('\n')
            refs = [l.split() for l in refs]

    scores = []
    for hyp, ref in zip(hyps, refs):
        scores.append(wpa(hyp, ref))

    return numpy.mean(scores)


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


def imt_f1_from_files(hyp_file, ref_file):
    """
    Compute average IMT F1 over the provided files
    :param hyp_file:
    :param ref_file:
    :return:

    Note: input files are assumed to be whitespace-tokenized, if this isn't the case this metric will be wrong

    """

    with codecs.open(hyp_file, encoding='utf8') as hyp_f:
        hyps = hyp_f.read().strip().split('\n')
        hyps = [l.split() for l in hyps]

        with codecs.open(ref_file, encoding='utf8') as ref_f:
            refs = ref_f.read().strip().split('\n')
            refs = [l.split() for l in refs]

    all_scores = []
    for hyp, ref in zip(hyps, refs):
        all_scores.append(imt_f1(hyp, ref))

    scores, p, r = zip(*all_scores)
    return numpy.mean(scores), numpy.mean(p), numpy.mean(r)

def imt_ndcg_from_files(hyp_file, ref_file):
    """
    Compute average IMT F1 over the provided files
    :param hyp_file:
    :param ref_file:
    :return:

    Note: input files are assumed to be whitespace-tokenized, with n-best
    lists separated by newlines, if this isn't the case this metric will be wrong
    """

    with codecs.open(hyp_file, encoding='utf8') as hyp_f:
        hyp_chunks = hyp_f.read().strip().split('\n\n')
        hyp_nbest_lists = [[l.split() for l in nbest_list.split('\n')] for nbest_list in hyp_chunks]

        with codecs.open(ref_file, encoding='utf8') as ref_f:
            ref_chunks = ref_f.read().strip().split('\n\n')
            ref_nbest_lists = [[l.split() for l in nbest_list.split('\n')] for nbest_list in ref_chunks]

    all_scores = []
    for hyps, refs in zip(hyp_nbest_lists, ref_nbest_lists):
        assert len(hyps) == len(refs), 'hyp and ref list lengths must match'
        chunk_scores = [imt_f1(hyp, ref) for hyp,ref in zip(hyps, refs)]
        chunk_f1s, chunk_p, chunk_r = zip(*chunk_scores)
        all_scores.append(ndcg(chunk_f1s))

    return numpy.mean(all_scores)

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
    actual_dcg = dcg(scores)
    if ideal_dcg == 0.0 or actual_dcg == 0.0:
        normed_cdg = 0.
    else:
        normed_cdg = actual_dcg / ideal_dcg
    return normed_cdg


