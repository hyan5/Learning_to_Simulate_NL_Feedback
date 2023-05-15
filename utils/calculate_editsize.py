from util_tools import load_json
from util_tools import store_json
from util_tools import edit_size
from argparse import ArgumentParser
from tqdm import tqdm
import sys
from collections import Counter

parser = ArgumentParser(description="Calculating the edit distance and adding to the output file.")
parser.add_argument("--input", '-i', type=str, nargs=1, help='the input file path')
parser.add_argument("--output", '-o', type=str, nargs=1, help='the output file path')

args = sys.argv
args, _ = parser.parse_known_args(args)


if __name__ == "__main__":
    input = args.input[0]
    output = args.output[0]

    input = load_json(input)
    counter = Counter()

    for sample in tqdm(input):
        gold_sql = sample['gold_parse']
        pred_sql = sample['predicted_parse_with_values']
        db_id = sample['db_id']
        
        editsize = edit_size(pred_sql, gold_sql, db_id)

        sample['editsize'] = editsize
        counter[editsize] += 1

    store_json(input, output)


    print("done!")
    print(f"{len(input)} samples in total.")
    print(f"Distribution: ")

    keys = list(counter.keys())
    #keys = list(counter.keys()).sort(reverse=True)
    keys.sort(reverse = True)
    for key in keys:
        print("EditSize: ", key, "\tFrequency: ", counter[key])
