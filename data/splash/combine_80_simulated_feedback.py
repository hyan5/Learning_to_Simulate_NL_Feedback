import json
import pdb

def load_json(file_path):
    return json.load(open(file_path, 'r'))

def read_file(file_path):
    results = []
    with open(file_path, 'r') as f:
        results = [line.strip() for line in f.readlines()]
    return results

def write_to_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    train_20 = load_json('train_20.json')
    train_80 = load_json('train_80.json')
    _train_80 = train_80
    train_80_sim_feedback = read_file('train_80_sim_feedback.sim')
    for data, fed in zip(train_80, train_80_sim_feedback):
        data['feedback'] = fed
    train_20 += train_80
    write_to_json(train_20, 'train_20_80_w_simulated_feedback.json')
    write_to_json(train_80, 'train_80_w_simulated_feedback.json')