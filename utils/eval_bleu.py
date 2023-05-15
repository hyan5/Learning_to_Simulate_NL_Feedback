import sys
sys.path.append('./../')

from config import SPLASH_TRAIN_JSON
from config import SPLASH_DEV_JSON
from config import SPLASH_TEST_JSON
import config
from parse_edit import Mappings
from parse_edit import EditParser
from utils_ import load_json
from utils_ import get_splash_key
from utils_ import parse_sql
from utils_ import eval_hardness
from utils_ import coverage
from os.path import join
import sys
from nltk.translate.bleu_score import corpus_bleu as nltk_bleu_corpus
from nltk.translate.bleu_score import SmoothingFunction
from sacrebleu import corpus_bleu as sacre_bleu_corpus
from sacrebleu import sentence_bleu as sacre_bleu
from fairseq.data.encoders.gpt2_bpe import get_encoder
from config import ENCODER
from config import VOCAB
from argparse import ArgumentParser
import os

global parser

def init():
    global parser
    parser = ArgumentParser(description='Calcuate the BLEU scores for simulated user feedbacks', allow_abbrev=False)
    parser.add_argument('--train_sim', default=config.SPLASH_TRAIN_SIM)
    parser.add_argument('--dev_sim', default=config.SPLASH_DEV_SIM)
    parser.add_argument('--train_ref', default=join(config.SPLASH_DIR, 'train.target'))
    parser.add_argument('--dev_ref', default=join(config.SPLASH_DIR, 'dev.target'))
    parser.add_argument('--log', default=config.LOG_FILE)

def eval_coverage(argv):
    global parser
    args, _ = parser.parse_known_args(argv)

    train_json = load_json(SPLASH_TRAIN_JSON)
    train_json += load_json(SPLASH_DEV_JSON)
    dev_json = load_json(SPLASH_TEST_JSON)

    with open(args.train_sim, 'r') as f:
        train_sims = f.readlines() 
    with open(args.dev_sim, 'r') as f:
        dev_sims = f.readlines()

    covereds = 0
    totals = 0

    trainCovereds = 0
    trainTotals = 0
    
    devCovereds = 0
    devTotals = 0

    trainEasyCovereds = 0
    trainMediumCovereds = 0
    trainHardCovereds = 0
    trainExtraCovereds = 0

    trainEasyTotals = 0
    trainMediumTotals = 0
    trainHardTotals = 0
    trainExtraTotals = 0

    devEasyCovereds = 0
    devMediumCovereds = 0
    devHardCovereds = 0
    devExtraCovereds = 0

    devEasyTotals = 0
    devMediumTotals = 0
    devHardTotals = 0
    devExtraTotals = 0

    for id, train in enumerate(train_json):
        goldSql = train['gold_parse']
        predSql = train['predicted_parse_with_values']
        db_id = train['db_id']
        feedback = train_sims[id]

        goldTree, schema, tables = parse_sql(goldSql, db_id)
        predTree, schema, tables = parse_sql(predSql, db_id)

        hardness = eval_hardness(goldTree)

        mappings = Mappings(schema)
        parser = EditParser(mappings)

        goldArgs = parser.parse_sql_tree(goldTree)
        predArgs = parser.parse_sql_tree(predTree)

        edits = parser.calEdits(predArgs, goldArgs)
        editsLin = parser.linearize(edits)
        # editSize = parser.editSize(edits)
        # TODO: calculate the coverage per editSize

        covered, total = coverage(editsLin, feedback)

        totals += total
        covereds += covered

        trainCovereds += covered
        trainTotals += total

        if hardness == 'easy':
            trainEasyCovereds += covered
            trainEasyTotals += total
        elif hardness == 'medium':
            trainMediumCovereds += covered
            trainMediumTotals += total
        elif hardness == 'hard':
            trainHardCovereds += covered
            trainHardTotals += total
        elif hardness == 'extra':
            trainExtraCovereds += covered
            trainExtraTotals += total

    for id, dev in enumerate(dev_json):
        goldSql = dev['gold_parse']
        predSql = dev['predicted_parse_with_values']
        db_id = dev['db_id']
        feedback = dev_sims[id]

        goldTree, schema, tables = parse_sql(goldSql, db_id)
        predTree, schema, tables = parse_sql(predSql, db_id)

        hardness = eval_hardness(goldTree)

        mappings = Mappings(schema)
        parser = EditParser(mappings)

        goldArgs = parser.parse_sql_tree(goldTree)
        predArgs = parser.parse_sql_tree(predTree)

        edits = parser.calEdits(predArgs, goldArgs)
        editsLin = parser.linearize(edits)
        # editSize = parser.editSize(edits)
        # TODO: calculate the coverage per editSize

        covered, total = coverage(editsLin, feedback)

        totals += total
        covereds += covered

        devCovereds += covered
        devTotals += total

        if hardness == 'easy':
            devEasyCovereds += covered
            devEasyTotals += total
        elif hardness == 'medium':
            devMediumCovereds += covered
            devMediumTotals += total
        elif hardness == 'hard':
            devHardCovereds += covered
            devHardTotals += total
        elif hardness == 'extra':
            devExtraCovereds += covered
            devExtraTotals += total

    all = covereds / totals

    train = trainCovereds / trainTotals
    dev = devCovereds / devTotals

    trainEasy = trainEasyCovereds / trainEasyTotals
    trainMedium = trainMediumCovereds / trainMediumTotals
    trainHard = trainHardCovereds / trainHardTotals
    trainExtra = trainExtraCovereds / trainExtraTotals

    devEasy = devEasyCovereds / devEasyTotals
    devMedium = devMediumCovereds / devMediumTotals
    devHard = devHardCovereds / devHardTotals
    devExtra = devExtraCovereds / devExtraTotals

    easy = (trainEasyCovereds + devEasyCovereds) / (trainEasyTotals + devEasyTotals)
    medium = (trainMediumCovereds + devMediumCovereds) / (trainMediumTotals + devMediumTotals)
    hard = (trainHardCovereds + devHardCovereds) / (trainHardTotals + devHardTotals)
    extra = (trainExtraCovereds + devExtraCovereds) / (trainExtraTotals + devExtraTotals)

    print(f'Average coverage: {all}')
    os.system(f'echo "Average coverage: {all}" >> {args.log}')
    print(f'Train coverage: {train}')
    os.system(f'echo "Train coverage: {train}" >> {args.log}')
    print(f'Dev coverage: {dev}')
    os.system(f'echo "Dev coverage: {dev}" >> {args.log}')
    print(f'trainEasy coverage: {trainEasy}')
    os.system(f'echo "trainEasy coverage: {trainEasy}" >> {args.log}')
    print(f'trainMedium coverage: {trainMedium}')
    os.system(f'echo "trainMedium coverage: {trainMedium}" >> {args.log}')
    print(f'trainHard coverage: {trainHard}')
    os.system(f'echo "trainHard coverage: {trainHard}" >> {args.log}')
    print(f'trainExtra coverage: {trainExtra}')
    os.system(f'echo "trainExtra coverage: {trainExtra}" >> {args.log}')
    print(f'devEasy coverage: {devEasy}')
    os.system(f'echo "devEasy coverage: {devEasy}" >> {args.log}')
    print(f'devMedium coverage: {devMedium}')
    os.system(f'echo "devMedium coverage: {devMedium}" >> {args.log}')
    print(f'devHard coverage: {devHard}')
    os.system(f'echo "devHard coverage: {devHard}" >> {args.log}')
    print(f'devExtra coverage: {devExtra}')
    os.system(f'echo "devExtra coverage: {devExtra}" >> {args.log}')
    print(f'easy coverage: {easy}')
    os.system(f'echo "easy coverage: {easy}" >> {args.log}')
    print(f'medium coverage: {medium}')
    os.system(f'echo "medium coverage: {medium}" >> {args.log}')
    print(f'hard coverage: {hard}')
    os.system(f'echo "hard coverage: {hard}" >> {args.log}')
    print(f'extra coverage: {extra}')
    os.system(f'echo "extra coverage: {extra}" >> {args.log}')

    return

def eval_bleu(argv):
    global parser
    args, _ = parser.parse_known_args(argv)
    print("Evaluation arguments: ", args)

    train_json = load_json(SPLASH_TRAIN_JSON)
    train_json += load_json(SPLASH_DEV_JSON)
    dev_json = load_json(SPLASH_TEST_JSON)

    train_ref = dict()
    for idx, item in enumerate(train_json):
        key = get_splash_key(item)
        if key in train_ref:
            train_ref[key]['refs'].append(item['feedback'])
            train_ref[key]['ref_idxs'].append(idx)
        else:
            train_ref[key] = {
                'db_id': item['db_id'], 
                'question': item['question'], 
                'gold_parse': item['gold_parse'],
                'predicted_parse': item['predicted_parse'],
                'predicted_parse_explanation': item['predicted_parse_explanation'],
                'refs': [item['feedback']],
                'ref_idxs': [idx]
            }

    dev_ref = dict()
    for idx, item in enumerate(dev_json):
        key = get_splash_key(item)
        if key in dev_ref:
            dev_ref[key]['refs'].append(item['feedback'])
            dev_ref[key]['ref_idxs'].append(idx)
        else:
            dev_ref[key] = {
                'db_id': item['db_id'], 
                'question': item['question'], 
                'gold_parse': item['gold_parse'],
                'predicted_parse': item['predicted_parse'],
                'predicted_parse_explanation': item['predicted_parse_explanation'],
                'refs': [item['feedback']],
                'ref_idxs': [idx]
            }

    train_cadidates = None
    with open(args.train_sim, 'r') as f:
        train_cadidates = f.readlines() 

    train_references = None
    with open(args.train_ref, 'r') as f:
        train_references = f.readlines()

    dev_cadidates = None
    with open(args.dev_sim, 'r') as f:
        dev_cadidates = f.readlines() 

    dev_references = None
    with open(args.dev_ref, 'r') as f:
        dev_references = f.readlines()

    # Calculate the nltk BLEU corpus score on the training set
    cads = []
    refss = []

    for key in train_ref:
        idxs = train_ref[key]['ref_idxs']
        cad = train_cadidates[idxs[0]].strip().split()
        cads.append(cad)
        refs = []
        for idx in idxs:
            ref = train_references[idx].strip().split()
            refs.append(ref)
        refss.append(refs)

    train_nltk_score = nltk_bleu_corpus(refss, cads)

    os.system(f'echo "NLTK BLEU score on training set: {train_nltk_score}" >> {args.log}')
    print(f'NLTK BLEU score on training set: {train_nltk_score}')

    # Calculate the nltk BLEU corpus score on the dev set
    cads = []
    refss = []

    for key in dev_ref:
        idxs = dev_ref[key]['ref_idxs']
        cad = dev_cadidates[idxs[0]].strip().split()
        cads.append(cad)
        refs = []
        for idx in idxs:
            ref = dev_references[idx].strip().split()
            refs.append(ref)
        refss.append(refs)

    dev_nltk_score = nltk_bleu_corpus(refss, cads)

    print(f'NLTK BLEU score on development set: {dev_nltk_score}')
    os.system(f'echo "NLTK BLEU score on development set: {dev_nltk_score}" >> {args.log}')

    # Calculate the sacre BLEU corpus score on the training set
    cads = []
    refss = []

    for key in train_ref:
        idxs = train_ref[key]['ref_idxs']
        cad = train_cadidates[idxs[0]].strip()
        cads.append(cad)
        refs = []
        for idx in idxs:
            ref = train_references[idx].strip()
            refs.append(ref)
        refss.append(refs)

    train_sacre_score = sacre_bleu_corpus(cads, refss)

    print(f'sacre BLEU score on training set: {train_sacre_score}')
    os.system(f'echo "sacre BLEU score on training set: {train_sacre_score}" >> {args.log}')

    # Calculate the sacre BLEU corpus score on the dev set
    cads = []
    refss = []

    for key in dev_ref:
        idxs = dev_ref[key]['ref_idxs']
        cad = dev_cadidates[idxs[0]].strip()
        cads.append(cad)
        refs = []
        for idx in idxs:
            ref = dev_references[idx].strip()
            refs.append(ref)
        refss.append(refs)

    dev_sacre_score = sacre_bleu_corpus(cads, refss)

    print(f'sacre BLEU score on development set: {dev_sacre_score}')
    os.system(f'echo "sacre BLEU score on development set: {dev_sacre_score}" >> {args.log}')

if __name__ == '__main__':
    init()
    argv = sys.argv
    eval_bleu(argv)
    eval_coverage(argv)
