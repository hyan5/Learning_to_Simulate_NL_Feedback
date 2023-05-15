from nltk import word_tokenize
import numpy as np
import json
import pdb

def read_file(file_dir):
    if 'json' in file_dir:
        return json.load(open(file_dir, 'r'))
    else:
        data = []
        with open(file_dir, 'r') as f:
            data = [line for line in f.readlines()]
        return data
    
if __name__ == '__main__':
    _train_src = read_file('data/splash/tqes/train.source')
    _train_tar = read_file('data/splash/tqes/train.target')
    train_src = [len(word_tokenize(line)) for line in _train_src]
    train_tar = [len(word_tokenize(line)) for line in _train_tar]

    print(f'Max Train Src: {max(train_src)}')
    print(f'Min Train Src: {min(train_src)}')
    print(f'Avg Train Src: {np.mean(np.array(train_src))}')

    print(f'Max Train Tar: {max(train_tar)}')
    print(f'Min Train Tar: {min(train_tar)}')
    print(f'Avg Train Tar: {np.mean(np.array(train_tar))}')

