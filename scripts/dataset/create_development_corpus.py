"""
Create a dev corpus for fast iteration on NMT research

Go through the big corpus and get N most frequent tokens

create the source and target vocabularies using these tokens

Select K segments which _only_ use tokens from the source and target vocabularies

write these K segments to train / dev files

We assume pre-tokenized data, this script is just for extracting good segments
"""

import logging
import argparse
import codecs
import itertools
import errno
import cPickle
from collections import Counter

from multiprocessing import Pool
import os

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source",
                    help="the source text corpus")
parser.add_argument("-t", "--target",
                    help="the target text corpus")
parser.add_argument("-v", "--vocab", type=int,
                    help="the vocabulary size")
parser.add_argument("-o", "--outputdir",
                    help="the directory where we should write the output")
parser.add_argument("-sl", "--sourcelang",  help="the source language code")
parser.add_argument("-tl", "--targetlang", help="the target language code")
parser.add_argument("-tk", "--traink", type=int, help="the number of training pairs you want")
parser.add_argument("-dk", "--devk", type=int, help="the number of dev pairs you want")



def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise



# 11000 segments, 10000 training, 1000 dev

def chunk_iterator(line_iterator, chunk_size=50000):
    while True:
        buffer = []
        for _ in range(chunk_size):
            try:
                buffer.append(line_iterator.next())
            except StopIteration:
                raise StopIteration
        yield buffer


def count_tokens(lines):
    src_toks = Counter()
    trg_toks = Counter()
    for source_line, target_line in lines:
        src_toks.update(source_line.strip().split())
        trg_toks.update(target_line.strip().split())

    return src_toks, trg_toks


def parallel_iterator(source_file, target_file):
    with codecs.open(source_file, encoding='utf8') as src:
        with codecs.open(target_file, encoding='utf8') as trg:
            for src_l, trg_l in itertools.izip(src, trg):
                yield (src_l, trg_l)


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dict = vars(args)

    chunk_factor = 500000
    parallel_chunk_iter = chunk_iterator(parallel_iterator(args.source, args.target), chunk_size=chunk_factor)

    num_procs = 10
    p = Pool(num_procs)

    source_vocab_counts = Counter()
    target_vocab_counts = Counter()
    # map and reduce
    for i, (src_count, trg_count) in enumerate(p.imap_unordered(count_tokens, parallel_chunk_iter)):
        source_vocab_counts += src_count
        target_vocab_counts += trg_count
        logger.info("{} lines processed".format(i*chunk_factor))

    source_vocab = [w for w, c in source_vocab_counts.most_common(args.vocab - 3)]
    target_vocab = [w for w, c in target_vocab_counts.most_common(args.vocab - 3)]

    # Special tokens and indexes
    default_mappings = [(u'<S>', 0), (u'<UNK>', 1), (u'</S>', args.vocab - 1)]
    source_vocab_dict = {k:v for k,v in default_mappings}
    target_vocab_dict = {k:v for k,v in default_mappings}

    for token, idx in zip(source_vocab, range(2, args.vocab - 1)):
        source_vocab_dict[token] = idx
    for token, idx in zip(target_vocab, range(2, args.vocab - 1)):
        target_vocab_dict[token] = idx

    min_len = 5
    max_len = 20
    max_len_diff = 4

    good_source_lines = []
    good_target_lines = []
    line_iter = parallel_iterator(args.source, args.target)
    count = 0
    while count < args.traink + args.devk:
        try:
            source_line, target_line = line_iter.next()
        except StopIteration:
            logger.error('I iterated through the whole dataset but there are not enough rows ' + \
                         'to get {} with your settings'.format(args.traink + args.devk))
            logger.error('You have {} lines'.format(count))
            break
        source_toks = source_line.strip().split()
        target_toks = target_line.strip().split()
        if (all(s_tok in source_vocab_dict for s_tok in source_toks) and
                all(t_tok in target_vocab_dict for t_tok in target_toks)):
            if min_len < len(source_toks) < max_len and min_len < len(target_toks) < max_len:
                if abs(len(source_toks) - len(target_toks)) < max_len_diff:
                    good_source_lines.append(source_line.strip())
                    good_target_lines.append(target_line.strip())
                    count += 1
                    if count % 100 == 0:
                        logger.info('Found {} lines so far'.format(count))

    source_train_lines = good_source_lines[:args.traink]
    target_train_lines = good_target_lines[:args.traink]
    source_dev_lines = good_source_lines[-args.devk:]
    target_dev_lines = good_target_lines[-args.devk:]

    # Now write the new files
    mkdir_p(args.outputdir)
    with codecs.open(os.path.join(args.outputdir, 'train.'+args.sourcelang), 'w', encoding='utf8') as out:
        for l in source_train_lines:
            out.write(l + u'\n')
    with codecs.open(os.path.join(args.outputdir, 'dev.'+args.sourcelang), 'w', encoding='utf8') as out:
        for l in source_dev_lines:
            out.write(l + u'\n')
    with codecs.open(os.path.join(args.outputdir, 'train.'+args.targetlang), 'w', encoding='utf8') as out:
        for l in target_train_lines:
            out.write(l + u'\n')
    with codecs.open(os.path.join(args.outputdir, 'dev.'+args.targetlang), 'w', encoding='utf8') as out:
        for l in target_dev_lines:
            out.write(l + u'\n')

    cPickle.dump(source_vocab_dict, open(os.path.join(args.outputdir, args.sourcelang + '.vocab.pkl'), 'w'))
    cPickle.dump(target_vocab_dict, open(os.path.join(args.outputdir, args.targetlang + '.vocab.pkl'), 'w'))

    logger.info('Wrote the development corpus to: {}'.format(args.outputdir))
    logger.info('Files in {}: {}'.format(args.outputdir, os.listdir(args.outputdir)))



