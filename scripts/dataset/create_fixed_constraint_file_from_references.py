"""
Go through a file of references, and generate constraints for each sequence

Use the random seed to ensure that generated constraints will always be the same
"""

import logging
import argparse
import codecs
import errno
import cPickle
import os

import numpy

from nn_imt.stream import n_constraints_from_sequence


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--references",
                    help="a file containing a list of references")
parser.add_argument("-o", "--outputdir",
                    help="the directory where we should write the output")
parser.add_argument("-tl", "--targetlang", help="the target language code")
parser.add_argument("-nc", "--maxnumconstraints", type=int, help="the maximum number of constraints per sequence")

RANDOM_SEED = 37
numpy.random.seed(RANDOM_SEED)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dict = vars(args)

    reference_lines = [l.strip().split()
                       for l in codecs.open(args.references, encoding='utf8').read().strip().split('\n')]

    all_constraints = []
    all_constraint_idxs = []
    for toks in reference_lines:
        # the empty array is just to conform to the signature used for NMT streams
        constraints, constraint_idx = n_constraints_from_sequence(numpy.array(toks), num_constraints=args.maxnumconstraints)

        all_constraints.append([c_tok for cons in constraints for c_tok in cons])
        all_constraint_idxs.append([c_idx for cons in constraint_idx for c_idx in cons])

    # Now write the new files
    mkdir_p(args.outputdir)
    with codecs.open(os.path.join(args.outputdir, 'constraints.'+args.targetlang), 'w', encoding='utf8') as out:
        for l in all_constraints:
            out.write(u' '.join(l) + u'\n')

    logger.info('Wrote constraints for {} to {}'.format(args.references, args.outputdir))

