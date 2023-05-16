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
    train_data = load_json('train_w_template_feedback.json')
    dev_data = load_json('dev_w_template_feedback.json')

    train_feedback = read_file('train.sim')
    dev_feedback = read_file('dev.sim')

    for i in range(len(train_data)):
        train_data[i]['feedback'] = train_feedback[i]
    
    for i in range(len(dev_data)):
        dev_data[i]['feedback'] = dev_feedback[i]

    write_to_json(train_data, 'train_w_simulated_feedback.json')
    write_to_json(dev_data, 'dev_w_simulated_feedback.json') 