import json
import random
from argparse import ArgumentParser
import pdb


parser = ArgumentParser(description="Preprocess the splash dataset into translation task format")
parser.add_argument("--percent", type=int, help='How many data of SPLASH train should be kept with real user feedback', default=20)
args = parser.parse_args()

def load_json(file_path):
    return json.load(open(file_path, 'r'))

def write_to_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    percent = args.percent / 100
    train_data = load_json('train_w_template_feedback.json')
    train_db = list(set([data['db_id'] for data in train_data]))
    split_eff = int(len(train_db) * percent)
    db_list = random.sample(train_db, split_eff)

    train_split1 = []
    train_split2 = []

    for i, data in enumerate(train_data):
        if data['db_id'] in db_list:
            train_split1.append(data)
        else:
            train_split2.append(data)
    pdb.set_trace()
    write_to_json(train_split1, f'train_{args.percent}.json')
    write_to_json(train_split2, f'train_{100-args.percent}.json')