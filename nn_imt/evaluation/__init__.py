import codecs
import numpy


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
    :param src_vocab:
    :param trg_vocab:
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

