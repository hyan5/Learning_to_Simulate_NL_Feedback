from utils.templates import StructureError, explanation, find_span
from util_tools import load_json
from util_tools import store_json
from argparse import ArgumentParser
from tqdm import tqdm
import config
import sys

parser = ArgumentParser(description="Generate explanation from the input file")
parser.add_argument("--input", '-i', type=str, nargs=1, help='the input file path')
parser.add_argument("--output", '-o', type=str, nargs=1, help='the output file path')
parser.add_argument("--no_nltk_tokenizer", action='store_true')

args = sys.argv
args, _ = parser.parse_known_args(args)
if args.no_nltk_tokenizer:
    config.NO_NLTK_TOKENIZER = True
config.VALUE_KEEP_QUOTE = False # Because the original explanation don't


def getInput(sample):
    # print(sample)
    if sample.get('prediction') is not None:
        field_name = 'prediction'
    else:
        field_name = "predicted_parse_with_values"
    return sample[field_name].split('|')[1].strip() if len(sample[field_name].split('|')) > 1 else ""

def generate_explanation(input, output):
    errors = []

    for sample in tqdm(input):
        gold_sql = getInput(sample)
        db_id = sample['db_id']
        
        try:
            fd = explanation(gold_sql, db_id)
        except:
            fd = []
            errors.append(sample)
        
        sample['generated_explanation'] = fd
    if output is not None:
        store_json(input, output)


    print("done!")
    print(f"{len(input)} samples in total.")
    print(f"{len(errors)} samples have empty explanation because uncovered pattern.")
    print(f"They are stored to _explanation_error_samples.json")

    if len(errors) > 0:
        store_json(errors, output.replace(".json", "_explanation_error_samples.json"))
    return input




if __name__ == "__main__":
    input = args.input[0]
    output = args.output[0]
    input = load_json(input)
    _ = generate_explanation(input, output)

    