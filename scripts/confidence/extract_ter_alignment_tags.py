import json
import sys
import codecs
import requests
import os
import subprocess
import argparse
import pdb

from corpus_generation import parse_pra
# extract the TER alignment from an n-best list generated from a list of source segments

#HYP_FILE='/media/1tb_drive/imt_models/EN-DE-confidence-baseline-PREDICTION_internal_data_dropout1.0_src_vocab80000_trg_vocab90000_emb300_rec1000_batch50/newstest2014.de.500.bpe.imt-hyps.out.10'
#REF_FILE='/media/1tb_drive/imt_models/EN-DE-confidence-baseline-PREDICTION_internal_data_dropout1.0_src_vocab80000_trg_vocab90000_emb300_rec1000_batch50/reference_suffixes.generated.10'

HYP_FILE='/media/1tb_drive/imt_models/EN-DE-confidence-baseline-PREDICTION_internal_data_dropout1.0_src_vocab80000_trg_vocab90000_emb300_rec1000_batch50/newstest2014.de.500.bpe.imt-hyps.out'
REF_FILE='/media/1tb_drive/imt_models/EN-DE-confidence-baseline-PREDICTION_internal_data_dropout1.0_src_vocab80000_trg_vocab90000_emb300_rec1000_batch50/reference_suffixes.generated'
output_dataset_prefix = '/media/1tb_drive/imt_models/EN-DE-confidence-baseline-PREDICTION_internal_data_dropout1.0_src_vocab80000_trg_vocab90000_emb300_rec1000_batch50/newstest2014.500.bpe_imt_test'

def get_lines(filename):
    with codecs.open(filename, encoding='utf8') as inp:
        return [l.split() for l in inp.read().strip().split('\n')]


hyp_lines = get_lines(HYP_FILE)
ref_lines = get_lines(REF_FILE)

assert len(hyp_lines) == len(ref_lines)

source_language = 'en'
target_language = 'de'

PATH_TERCOM='/home/chris/programs/tercom-0.7.25/tercom.7.25.jar'

# Create hypothesis and reference files for TERCOM.
hypothesis_file = '%s_%s-%s.tercom.hyp' % (output_dataset_prefix,
                                           source_language,
                                           target_language)
reference_file = '%s_%s-%s.tercom.ref' % (output_dataset_prefix,
                                          source_language,
                                          target_language)
with codecs.open(hypothesis_file, 'w', 'utf8') as f_hyp, codecs.open(reference_file, 'w', 'utf8') as f_ref:
    for i, (hyp, ref) in enumerate(zip(hyp_lines,
                                       ref_lines)):
        f_hyp.write('%s\t(%.12d)\n' % (hyp, i))
        f_ref.write('%s\t(%.12d)\n' % (ref, i))

# Run TERCOM.
output_prefix = '%s_%s-%s.tercom.out' % (output_dataset_prefix,
                                         source_language,
                                         target_language)
cmd = 'java -jar %s -r %s -h %s -n %s -d 0' % (PATH_TERCOM,
                                               reference_file,
                                               hypothesis_file,
                                               output_prefix)
p = subprocess.Popen(cmd, shell=True, stderr=sys.stderr, stdout=sys.stdout)
p.wait()

# Run Varvara's script to create OK/BAD tags.
parse_pra.parse_file('%s.pra' % output_prefix)
tag_file = '%s.pra.tags' % output_prefix

print("Wrote tags to: {}".format(tag_file))

import ipdb; ipdb.set_trace()

