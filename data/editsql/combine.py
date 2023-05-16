import json
import random
import pdb

def load_json(file_path):
    return json.load(open(file_path, 'r'))

def write_to_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    run1 = load_json('editsql_error/run_1_err.json')
    run2 = load_json('editsql_error/run_2_err.json')
    run3 = load_json('editsql_error/run_3_err.json')
    print(f'Total Len: {len(run1) + len(run2) + len(run3)}')
    test = load_json('splash-editsql.json')
    for i, exp in enumerate(test):
        test[i]['predicted_parse_with_values'] = test[i].pop('predicted_parse')
    run1 += run2
    run1 += run3
    print(f'Total Len: {len(run1)}')

    random.shuffle(run1)
    train_split = int(0.8 * len(run1))
    train_data = run1[:train_split]
    dev_data = run1[train_split:]
    write_to_json(train_data, 'train.json')
    write_to_json(dev_data, 'dev.json')
    write_to_json(test, 'test.json')

