import sys
from templates import StructureError, feedback, find_span, explanation
from argparse import ArgumentParser
from tqdm import tqdm
import config
from config import ADD_TAG1, ADD_TAG2, SUB_TAG1, SUB_TAG2, INFO_TAG1, INFO_TAG2
from unidecode import unidecode
from util_tools import load_json, cal_edits, store_json
from nltk import word_tokenize
import pdb

parser = ArgumentParser(description="Generate feedback from the input file")
parser.add_argument("--input", '-i', type=str, nargs=1, help='the input file path')
parser.add_argument("--output", '-o', type=str, nargs=1, help='the output file path')
parser.add_argument("--no_underscore", action='store_true', help="remove underscores in feedback")
parser.add_argument("--no_quote", action='store_true', help="remove quota from values in feedback")
parser.add_argument("--show_tag", action='store_true', help="add tags to feedback")
parser.add_argument("--connect_foreign_key_group", action='store_true', help="connect foreign key group in feedback")
parser.add_argument("--only_column_name", action='store_true', help="when comparing two columns in feedback, we only consider the column name")
parser.add_argument("--no_nltk_tokenizer", action='store_true', help="for not using NLTK tokenizer")
parser.add_argument("--verbal", action='store_true', help="print exception messages for structural errors")
parser.add_argument("--use_modified_schema", action='store_true', help="use modified schema names instead of their original names")


args = sys.argv
args, _ = parser.parse_known_args(args)

if args.no_underscore:
    config.REPLACE_UNDERSCORE_WITH_SPACE = True
if args.no_quote:
    config.VALUE_KEEP_QUOTE = False
if args.show_tag:
    config.SHOW_TAG = True
if args.connect_foreign_key_group:
    config.CONNECT_FOREIGN_KEY_GROUP = True
if args.only_column_name:
    config.ONLY_COLUMN_NAME = True
if args.no_nltk_tokenizer:
    config.NO_NLTK_TOKENIZER = True
if args.use_modified_schema:
    config.USE_MODIFIED_SCHEMA = True

def generate_feedback(input, output):
    filtered = []
    kept = []
    no_feedback = 0
    undefined_error = 0
    for sample in tqdm(input):
        gold_sql = unidecode(sample['gold_parse'])
        pred_sql = unidecode(sample['predicted_parse_with_values'])
        db_id = sample['db_id']
        
        try:
            fd = feedback(pred_sql, gold_sql, db_id, verbal=args.verbal)
            if len(fd) == 0 or fd[:7] == 'ERROR: ' :
                sample['error info'] = fd
                filtered += [sample]
                if len(fd) == 0: no_feedback += 1
                continue
        except Exception as e:
            # The exceptions captured here are undefined errors
            sample['error info'] = f'ERROR: (undefined_error) {repr(e)}'
            filtered += [sample]
            undefined_error += 1
            continue
        
        d, edit_size, _edits_no_tag, _edits = cal_edits(pred_sql, gold_sql, db_id)
        _d = ' '.join(word_tokenize(d)).replace("``", "\"").replace("''", "\"").replace("< /", "</")

        sample['template_feedback'] = fd

        try:
            sample['feedback'] = unidecode(sample['feedback'])
        except:
            pass
        
        sample['question'] = unidecode(sample['question'])

        original_show_tag = config.SHOW_TAG
        config.SHOW_TAG = True
        fd = feedback(pred_sql, gold_sql, db_id)
        add_span = find_span(ADD_TAG1, ADD_TAG2, [SUB_TAG1, SUB_TAG2], fd)
        sub_span = find_span(SUB_TAG1, SUB_TAG2, [ADD_TAG1, ADD_TAG2], fd)
        # info_span = find_span(INFO_TAG1, INFO_TAG2, fd)
        config.SHOW_TAG = original_show_tag

        exp = explanation(pred_sql, db_id)
        sample['predicted_parse_explanation'] = exp
        sample['primary_span'] = add_span
        sample['secondary_span'] = sub_span
        sample['edits'] = _edits_no_tag
        sample['edits_original'] = _d
        # sample['info_span'] = info_span
        kept.append(sample)
    if output is not None:
        store_json(kept, output)
        store_json(filtered, output.replace(".json", "_struct_err.json"))

    print("done!")
    print(f"{len(input)} samples in total; {len(kept)} are kept and {len(filtered)} are filtered out due some mistakes.")
    print(f'In all of the errors, {sum(config.ABANDONED.values())} samples are defined structural errors, {no_feedback} samples with no feedback because the parser think the the predicted SQL is right, {undefined_error} samples with undefined error.')
    print("Below are detailes of defined structural errors:")
    print(f"{config.ABANDONED[1]} have error feedback because trying to remove a union/intersect/except operation.")
    print(f"{config.ABANDONED[3]} have error feedback because trying to add a union/intersect/except operation.")
    print(f"{config.ABANDONED[6]} have error feedback because trying to remove a nested SQL query from FROM clause.")
    print(f"{config.ABANDONED[7]} have error feedback because trying to add a nested SQL query from FROM clause.")
    print(f"{config.ABANDONED[20]} have error feedback because trying to add a complex nested SQL query (multi-step) from WHERE clause.")
    print(f"{config.ABANDONED[21]} have error feedback because trying to remove a complex nested SQL query (multi-step) from WHERE clause.")
    # print(f"{config.ABANDONED[101]} have error feedback because trying to add/remove a nested SQL query from WHERE clause.")
    print(f"{config.ABANDONED[102]} have error feedback because trying to modify a nested complex SQL query with more than one tables from FROM clause.")
    print(f"{config.ABANDONED[103]} have error feedback because more than one subquires occured in one condition unit.")
    print(f"{config.ABANDONED[104]} have error feedback because more than one conditions in HAVING clause.")
    print(f"{config.ABANDONED[105]} have error feedback because nested SQL occured in HAVING clause.")
    print(f"{config.ABANDONED[106]} have error feedback because trying to add entire HAVING clause.")
    print(f"{config.ABANDONED[107]} have error feedback because trying to remove entire HAVING clause.")


if __name__ == "__main__":
    input = args.input[0]
    output = args.output[0]
    input = load_json(input)
    generate_feedback(input, output)

    
