from argparse import ArgumentParser
import pdb
import json
# import wandb
import os
import sys
import torch
import numpy as np
from nltk import word_tokenize
from evaluation import SpiderMetric
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
# os.environ["WANDB_MODE"]="offline"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
# wandb.init(project="feedback-correction", entity="hyan5")

parser = ArgumentParser()
parser.add_argument("--data_dir",type=str,required=True,
    help="Path of the dataset")

parser.add_argument("--data_revision",type=str,required=True,
    help="Revision of the dataset")

parser.add_argument("--model", type=str, required=True, 
    help="Model name, T5-base, T5-large, T5-3b")

parser.add_argument("--ckp", type=str, required=True, 
    help="Model name, T5-base, T5-large, T5-3b")

parser.add_argument("--local_rank", type=int, required=False, 
    help="Local rank")

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)

def read_data(data_path):
    results = []
    with open(data_path, "r") as f:
        results = [line for line in f.readlines()]
    return results

# def compute_metrics(eval_preds):
#     metric = evaluate.load('bleu')
#     predictions, labels = eval_preds
#     predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     labels = [[label] for label in tokenizer.batch_decode(labels, skip_special_tokens=True)]
#     return metric.compute(predictions=predictions, references=labels, tokenizer=word_tokenize, smooth=True)

def compute_metrics(eval_preds):
    eval_dir = 'evaluation_results'
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    metric = SpiderMetric(f'{eval_dir}/dev_gold.txt', f'{eval_dir}/{revision}_pred.txt', 'evaluation/database', 'evaluation/tables.json', 'all', plug_value=False)
    predictions, labels = eval_preds
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    with open(f'{eval_dir}/{revision}_pred.txt', 'w') as f:
        for pred in predictions:
            f.write('%s\n' %pred)
    scores = metric.run_evaluation()
    return scores

class SplashDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_inputs, encoded_labels, tokenizer):
        self.inputs = encoded_inputs
        self.labels = encoded_labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = {key: torch.LongTensor(val[idx]) for key, val in self.inputs.items()}
        label_ids = self.labels["input_ids"][idx]
        # label_ids[label_ids == self.tokenizer.pad_token_id] = -100 #pad token id (incomplete)
        item['labels'] = torch.LongTensor(label_ids)
        return item

    def __len__(self):
        return len(self.labels['input_ids'])

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = args.model
    data_dir = args.data_dir
    global revision
    revision = args.data_revision
    cache_dir = model_type.replace('-', '_')

    max_source_length = 1024
    max_target_length = 256

    train_src = read_data(os.path.join(f'{data_dir}_{revision}', 'train.source'))
    train_tar = read_data(os.path.join(f'{data_dir}_{revision}', 'train.target'))

    dev_src = read_data(os.path.join(f'{data_dir}_{revision}', 'dev.source'))
    dev_tar = read_data(os.path.join(f'{data_dir}_{revision}', 'dev.target'))

    test_src = read_data(os.path.join(f'{data_dir}_{revision}', 'test.source'))
    test_tar = read_data(os.path.join(f'{data_dir}_{revision}', 'test.target'))

    additional_special_tokens = ['[question]', '[schema]', '[T]', '[C]', '[true]', '[predict]', '[feedback]', '[system description]']

    model = AutoModelForSeq2SeqLM.from_pretrained(f'results-{model_type}-{revision}/checkpoint-{args.ckp}', cache_dir=cache_dir)

    tokenizer.add_tokens(additional_special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    train_encodings = tokenizer(train_src, truncation=True, padding="longest", max_length=max_source_length)
    train_labels_encodings = tokenizer(train_tar, truncation=True, padding="longest", max_length=max_target_length)
    dev_encodings = tokenizer(dev_src, truncation=True, padding="longest", max_length=max_source_length)
    dev_labels_encodings = tokenizer(dev_tar, truncation=True, padding="longest", max_length=max_target_length)
    test_encodings = tokenizer(test_src, truncation=True, padding="longest", max_length=max_source_length)
    test_labels_encodings = tokenizer(test_tar, truncation=True, padding="longest", max_length=max_target_length)

    train_dataset = SplashDataset(train_encodings, train_labels_encodings, tokenizer)
    dev_dataset = SplashDataset(dev_encodings, dev_labels_encodings, tokenizer)
    test_dataset = SplashDataset(test_encodings, test_labels_encodings, tokenizer)

    # pdb.set_trace()
    # for data in train_dataset:
    #     pdb.set_trace()

    training_args = Seq2SeqTrainingArguments(
        output_dir=f'./results-{model_type}-{revision}-train',
        overwrite_output_dir=True,       
        num_train_epochs=500,             
        per_device_train_batch_size=5,
        per_device_eval_batch_size=5,  
        warmup_steps=0,
        weight_decay=0.01,          
        logging_dir='./logs',  
        logging_steps=4,
        evaluation_strategy="steps",
        eval_steps=64,
        learning_rate=1e-4,
        predict_with_generate=True,
        save_strategy='steps',
        save_steps=64,
        save_total_limit=128,
        metric_for_best_model = 'exact_score',
        load_best_model_at_end=True,
        gradient_accumulation_steps=64,
        lr_scheduler_type="constant",
        logging_first_step=True,
        # report_to=['wandb'],
        # deepspeed='./deepspeed-zero2.json',
    )
    # training_args.pop("local_rank")
    # print(training_args)
    trainer = Seq2SeqTrainer(
        model=model,                        
        args=training_args,                  
        train_dataset=train_dataset,       
        eval_dataset=train_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # trainer.train()
    # trainer.evaluate()
    # wandb.finish()
    trainer.save_model()

    preds, label_ids, metrics = trainer.predict(test_dataset=dev_dataset, max_length=max_target_length)
    predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
    with open(f'./results-{model_type}-{revision}-train/dev.sim', 'w') as f:
        for pred in predictions:
            f.write("%s\n" % pred)
    # preds, label_ids, metrics = trainer.predict(test_dataset=test_dataset, max_length=max_target_length)
    # predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # with open(f'./results-{model_type}-{revision}/test.sim', 'w') as f:
    #     for pred in predictions:
    #         f.write("%s\n" % pred)