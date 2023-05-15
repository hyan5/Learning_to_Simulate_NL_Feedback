# coding: utf-8
import logging
import sys

import random
import os
import json
# import pdb

from tqdm import tqdm
import numpy as np
import torch
import wandb
import copy

from data import SpiderAlignDataset
from aligner_model import BertAlignerModel
from utils.utils import AverageMeter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['WANDB_MODE'] = 'offline'


lr = 'roberta'
blr = 1e-8
l1 = 0.1
prior_pos = 1e-3
prior_neg = 1e-3
cls_weight = 0
mar = 0.1
epochs = 200
model_shortcut = 'roberta-large'

# model_shortcut = 'bert-base-uncased'


batch_size = 64

replace_mode = 'replace'
n_random = 0
n_changing = 50
n_swapping = 0
n_dropping = 0

pre = f'{n_random}-{n_changing}-{n_swapping}-{n_dropping}'

similarity_mode = 'average'

date = '12/21'

wandb_config = {
  "aligner_lr": lr,
  "bert_lr": blr,
  "epochs": epochs,
  "batch_size": batch_size,
  "margin": mar,
  "l1-norm_weight": l1,
  "cls": cls_weight,
  "pos_prior": prior_pos,
  "neg_prior": prior_neg,
  "negative-mode": f'{replace_mode}-{n_random}-{n_changing}-{n_swapping}-{n_dropping}',
  "similarity_mode": similarity_mode
}
wandb.init(project="feedback-evaluation", entity='hyan5', config=wandb_config)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

# What are the names of colleges that have two or more players, listed in descending alphabetical order?
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(1229)
torch.manual_seed(1229)
torch.cuda.manual_seed(1229)

def train(model, dataloader, criterion, optimizer):
    global total_training_iter
    model.train()
    bat_loss = []
    bat_pure_loss = []
    bat_l1_term = []
    with tqdm(dataloader) as tqdm_dataloader:
        average_meter = AverageMeter()

        for batch_data in tqdm_dataloader:

            tensors, weights, lengths, texts, att_weights, schemas, schema_idx, pos_prior, neg_prior = batch_data

            template_tensors, positive_tensors, negative_tensors = tensors
            template_weights, positive_weights, negative_weights = weights
            template_lengths, positive_lengths, negative_lengths = lengths
            template_texts, positive_texts, negative_texts = texts
            template_att_weights, positive_att_weights, negative_att_weights = att_weights
            template_schema, positive_schema, negative_schema = schemas
            template_schema_idx, positive_schema_idx, negative_schema_idx = schema_idx
            prior_temp_pos_mat, prior_temp_pos_mask_mat = pos_prior
            prior_temp_neg_mat, prior_temp_neg_mask_mat = neg_prior

            template_tensors, positive_tensors, negative_tensors = template_tensors.to(device), positive_tensors.to(device), negative_tensors.to(device)
            template_weights, positive_weights, negative_weights = template_weights.to(device), positive_weights.to(device), negative_weights.to(device)
            template_lengths, positive_lengths, negative_lengths = template_lengths.to(device), positive_lengths.to(device), negative_lengths.to(device)
            template_att_weights, positive_att_weights, negative_att_weights = template_att_weights.to(device), positive_att_weights.to(device), negative_att_weights.to(device)
            

            prior_temp_pos_mat, prior_temp_pos_mask_mat = prior_temp_pos_mat.to(device), prior_temp_pos_mask_mat.to(device)
            prior_temp_neg_mat, prior_temp_neg_mask_mat =prior_temp_neg_mat.to(device), prior_temp_neg_mask_mat.to(device)
            

            batch_size = template_tensors.size(0)

            temp_max_len = torch.LongTensor([template_lengths.max()]).expand(batch_size, 1)
            pos_max_len = torch.LongTensor([positive_lengths.max()]).expand(batch_size, 1)
            neg_max_len = torch.LongTensor([negative_lengths.max()]).expand(batch_size, 1)
            
            # positive_similar_matrix, negative_similar_matrix = model(positive_tensors, positive_lengths, negative_tensors, negative_lengths)
            if (not isinstance(model, torch.nn.DataParallel) and model.use_autoencoder) or \
                    (isinstance(model, torch.nn.DataParallel) and model.module.use_autoencoder):
                positive_similar_matrix, positive_cls_similarity, negative_similar_matrix, negative_cls_similarity, autoencoder_diff = \
                    model(template_tensors, template_lengths, template_weights, template_att_weights,
                          positive_tensors, positive_lengths, positive_weights, positive_att_weights,
                          negative_tensors, negative_lengths, negative_weights, negative_att_weights,
                          temp_max_len, pos_max_len, neg_max_len, mode='train')
            else:
                positive_similar_matrix, positive_cls_similarity, negative_similar_matrix, negative_cls_similarity = \
                    model(template_tensors, template_lengths, template_weights, template_att_weights,
                          positive_tensors, positive_lengths, positive_weights, positive_att_weights,
                          negative_tensors, negative_lengths, negative_weights, negative_att_weights,
                          temp_max_len, pos_max_len, neg_max_len, mode='train')
                autoencoder_diff = None


            if torch.cuda.is_available():
                positive_lengths = positive_lengths.cuda()
                negative_lengths = negative_lengths.cuda()
            # pdb.set_trace()
            loss, pure_loss, l1_term, pos_prior_loss, neg_prior_loss = criterion(positive_similar_matrix, negative_similar_matrix, (positive_cls_similarity, negative_cls_similarity), cls_weight, (template_lengths, positive_lengths, negative_lengths), (prior_temp_pos_mat, prior_temp_pos_mask_mat), (prior_temp_neg_mat, prior_temp_neg_mask_mat))
            
            # pdb.set_trace()
            if autoencoder_diff:
                loss = loss + autoencoder_diff
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            average_meter.update(loss.item(), pure_loss.item(), l1_term.item(), pos_prior_loss.item(), neg_prior_loss.item())
            tqdm_dataloader.set_postfix_str('loss = {:.4f}'.format(average_meter.hinge_loss_avg))

            bat_loss.append(loss.item())
            bat_pure_loss.append(pure_loss.item())
            bat_l1_term.append(l1_term.item())

        return average_meter.hinge_loss_avg, average_meter.pure_loss_avg, average_meter.l1_term_avg, average_meter.pos_prior_avg, average_meter.neg_prior_avg, bat_loss, bat_pure_loss, bat_l1_term


def validate(model, dataloader, criterion, neg_len, neg_tag, epoch, data_type, out_format, similarity_mode):
    model.eval()
    bat_loss = []
    bat_pure_loss = []
    bat_l1_term = []
    bat_coverage_term = []
    with tqdm(dataloader) as tqdm_dataloader:
        average_meter = AverageMeter()

        all_temp_lens, all_pos_lens, all_neg_lens = [], [], []
        all_pos_alignments, all_neg_alignments = [], []

        # indicator = 0
        for batch_data in tqdm_dataloader:
            # if indicator > 2:
            #     break
            # indicator += 1
            tensors, weights, lengths, texts, att_weights, schemas, schema_idx, pos_prior, neg_prior = batch_data
            template_tensors, positive_tensors, negative_tensors = tensors
            template_weights, positive_weights, negative_weights = weights
            template_lengths, positive_lengths, negative_lengths = lengths
            template_texts, positive_texts, negative_texts = texts
            template_att_weights, positive_att_weights, negative_att_weights = att_weights
            template_schema, positive_schema, negative_schema = schemas
            template_schema_idx, positive_schema_idx, negative_schema_idx = schema_idx
            prior_temp_pos_mat, prior_temp_pos_mask_mat = pos_prior
            prior_temp_neg_mat, prior_temp_neg_mask_mat = neg_prior

            template_tensors, positive_tensors, negative_tensors = template_tensors.to(device), positive_tensors.to(device), negative_tensors.to(device)
            template_weights, positive_weights, negative_weights = template_weights.to(device), positive_weights.to(device), negative_weights.to(device)
            template_lengths, positive_lengths, negative_lengths = template_lengths.to(device), positive_lengths.to(device), negative_lengths.to(device)
            template_att_weights, positive_att_weights, negative_att_weights = template_att_weights.to(device), positive_att_weights.to(device), negative_att_weights.to(device)
            prior_temp_pos_mat, prior_temp_pos_mask_mat = prior_temp_pos_mat.to(device), prior_temp_pos_mask_mat.to(device)
            prior_temp_neg_mat, prior_temp_neg_mask_mat =prior_temp_neg_mat.to(device), prior_temp_neg_mask_mat.to(device)
            
            batch_size = template_tensors.size(0)

            temp_max_len = torch.LongTensor([template_lengths.max()]).expand(batch_size, 1)
            pos_max_len = torch.LongTensor([positive_lengths.max()]).expand(batch_size, 1)
            neg_max_len = torch.LongTensor([negative_lengths.max()]).expand(batch_size, 1)
            # positive_similar_matrix, negative_similar_matrix = \
            #     model(positive_tensors, positive_lengths, negative_tensors, negative_lengths)
            # pdb.set_trace()

            if (not isinstance(model, torch.nn.DataParallel) and model.use_autoencoder) or \
                    (isinstance(model, torch.nn.DataParallel) and model.module.use_autoencoder):
                positive_similar_matrix, positive_cls_similarity, negative_similar_matrix, negative_cls_similarity, autoencoder_diff = \
                    model(template_tensors, template_lengths, template_weights, template_att_weights,
                          positive_tensors, positive_lengths, positive_weights, positive_att_weights,
                          negative_tensors, negative_lengths, negative_weights, negative_att_weights,
                          temp_max_len, pos_max_len, neg_max_len, mode='train')
            else:
                positive_similar_matrix, positive_cls_similarity, negative_similar_matrix, negative_cls_similarity = \
                    model(template_tensors, template_lengths, template_weights, template_att_weights,
                          positive_tensors, positive_lengths, positive_weights, positive_att_weights,
                          negative_tensors, negative_lengths, negative_weights, negative_att_weights,
                          temp_max_len, pos_max_len, neg_max_len, mode='train')
                autoencoder_diff = None
            if torch.cuda.is_available():
                positive_lengths = positive_lengths.cuda()
                negative_lengths = negative_lengths.cuda()

            loss, pure_loss, l1_term, pos_prior_loss, neg_prior_loss = criterion(positive_similar_matrix, negative_similar_matrix, (positive_cls_similarity, negative_cls_similarity), cls_weight, (template_lengths, positive_lengths, negative_lengths), (prior_temp_pos_mat, prior_temp_pos_mask_mat), (prior_temp_neg_mat, prior_temp_neg_mask_mat))
            
            bat_loss.append(loss.item())
            bat_pure_loss.append(pure_loss.item())
            bat_l1_term.append(l1_term.item())

            average_meter.update(loss.item(), pure_loss.item(), l1_term.item(), pos_prior_loss.item(), neg_prior_loss.item())
            tqdm_dataloader.set_postfix_str('loss = {:.4f}'.format(average_meter.hinge_loss_avg))
            
            temp_lens = template_lengths.squeeze().cpu().numpy().tolist()
            pos_lens = positive_lengths.squeeze().cpu().numpy().tolist()
            neg_lens = negative_lengths.squeeze().cpu().numpy().tolist()
            pos_aligns = positive_similar_matrix.detach().cpu().numpy().tolist()
            neg_aligns = negative_similar_matrix.detach().cpu().numpy().tolist()

            all_temp_lens += temp_lens if isinstance(temp_lens, list) else [temp_lens]
            all_pos_lens += pos_lens if isinstance(pos_lens, list) else [pos_lens]
            all_neg_lens += neg_lens if isinstance(neg_lens, list) else [neg_lens]
            all_pos_alignments += pos_aligns if len(pos_aligns) != 1 else [pos_aligns]
            all_neg_alignments += neg_aligns if len(neg_aligns) != 1 else [neg_aligns]
    alignments = [[np.array(_) for _ in all_pos_alignments], [np.array(_) for _ in all_neg_alignments]]

    lengths = [np.array(all_temp_lens), np.array(all_pos_lens), np.array(all_neg_lens)]

    val_acc_all, val_acc_changing, val_acc_swapping, val_acc_dropping, val_acc_random= validate_acc(alignments, lengths, neg_len, neg_tag, epoch, data_type, out_format, similarity_mode)
    # print(f'Validate acc = {val_acc}')
    return average_meter.hinge_loss_avg, average_meter.pure_loss_avg, average_meter.l1_term_avg, average_meter.pos_prior_avg, average_meter.neg_prior_avg, val_acc_all, val_acc_changing, val_acc_swapping, val_acc_dropping, val_acc_random


def validate_acc(alignments, lengths, masked, neg_tag, epoch, data_type, out_format, similarity_mode):
    """ Validate accuracy: whether model can choose the positive
    sentence over other negative samples """
    pos_scores, neg_scores = [], []
    pos_alignments, neg_alignments = alignments
    src_lengths, pos_tgt_lengths, neg_tgt_lengths = lengths

    assert len(pos_alignments) == len(neg_alignments) == len(src_lengths) == len(pos_tgt_lengths) == len(neg_tgt_lengths)

    for pos_alignment, neg_alignment, src_len, pos_tgt_len, neg_tgt_len \
            in zip(pos_alignments, neg_alignments, src_lengths, pos_tgt_lengths, neg_tgt_lengths):
        # print(np.shape(pos_alignment))
        # print(src_len)
        # print(pos_tgt_len, neg_tgt_len)
        if similarity_mode == 'recall':
            # pdb.set_trace()
            pos_score = np.sum(pos_alignment.max(1)) / src_len
            neg_score = np.sum(neg_alignment.max(1)) / src_len
        elif similarity_mode == 'precision':
            pos_score = np.sum(pos_alignment.max(0)) / pos_tgt_len
            neg_score = np.sum(neg_alignment.max(0)) / neg_tgt_len
        elif similarity_mode == 'f1':
            pos_recall_score, pos_precision_score = np.sum(pos_alignment.max(1)) / src_len, np.sum(pos_alignment.max(0)) / pos_tgt_len
            neg_recall_score, neg_precision_score = np.sum(neg_alignment.max(1)) / src_len, np.sum(neg_alignment.max(0)) / neg_tgt_len

            pos_score = 2 * pos_recall_score * pos_precision_score / (pos_recall_score + pos_precision_score)
            neg_score = 2 * neg_recall_score * neg_precision_score / (neg_recall_score + neg_precision_score)
        else:
            pos_recall_score, pos_precision_score = np.sum(pos_alignment.max(1)) / src_len, np.sum(pos_alignment.max(0)) / pos_tgt_len
            neg_recall_score, neg_precision_score = np.sum(neg_alignment.max(1)) / src_len, np.sum(neg_alignment.max(0)) / neg_tgt_len
            pos_score = (pos_recall_score + pos_precision_score) / 2
            neg_score = (neg_recall_score + neg_precision_score) / 2

        pos_scores.append(pos_score)
        neg_scores.append(neg_score)
    # print("Pos Scores: ", pos_scores)
    # print("Neg Scores: ", neg_scores)
    num_examples, num_corrects = 0, 0
    new_num_examples, new_num_corrects = 0, 0
    ranking_sum = 0.0

    num_changing_examples, num_changing_corrects = 0, 0
    new_num_changing_examples, new_num_changing_corrects = 0, 0
    ranking_changing = [] 

    num_swapping_examples, num_swapping_corrects = 0, 0
    new_num_swapping_examples, new_num_swapping_corrects = 0, 0
    ranking_swapping = [] 

    num_dropping_examples, num_dropping_corrects = 0, 0
    new_num_dropping_examples, new_num_dropping_corrects = 0, 0
    ranking_dropping = []

    num_random_examples, num_random_corrects = 0, 0
    new_num_random_examples, new_num_random_corrects = 0, 0
    ranking_random = [] 

    neg_len_pointer = 0
    itr_ind = 0

    # print('len_pos_scores: ', len(pos_scores)/neg_sample_num)
    # print('len_neg_scores: ', len(neg_scores)/neg_sample_num)
    # print('len_masked: ', len(masked))
    scores = []

    bat_acc_new = []
    for i in range(0, len(pos_scores), batch_size):
        total_count = 0
        correct_count = 0
        for bat in range(min(batch_size, len(pos_scores)- i)):
            total_count += 1
            if pos_scores[i+bat] > neg_scores[i+bat]:
                correct_count += 1
        bat_acc_new.append(1. * correct_count / total_count)


    while neg_len_pointer < len(pos_scores):
        # print("Iteration Indicator: ", itr_ind)
        neg_len = masked[itr_ind]
        one_pos_scores = pos_scores[neg_len_pointer: neg_len_pointer + neg_len]
        one_neg_scores = neg_scores[neg_len_pointer: neg_len_pointer + neg_len]
        num_examples += 1

        num_phrase_changing, num_phrase_swapping, num_phrase_dropping, num_random = neg_tag[itr_ind]

        
        # MRR calculation
        rank_score_list = copy.deepcopy(one_neg_scores)
        rank_score_list.append(one_pos_scores[0])
        rankings = sorted(rank_score_list, reverse=True)
        index = rankings.index(one_pos_scores[0])
        ranking_sum += 1. / (index+1)

        if one_pos_scores[0] > max(one_neg_scores):
            num_corrects += 1

        for j in range(len(one_pos_scores)):
            new_num_examples += 1
            if one_pos_scores[j] > one_neg_scores[j]:
                new_num_corrects += 1
        neg_len_pointer += neg_len
        itr_ind += 1

        if num_phrase_changing != 0:
            neg_phrase_changing_scores = one_neg_scores[: num_phrase_changing]
            num_changing_examples += 1

            for j, score in enumerate(neg_phrase_changing_scores):
                new_num_changing_examples += 1
                if one_pos_scores[0] > score:
                    new_num_changing_corrects += 1

            if one_pos_scores[0] > max(neg_phrase_changing_scores):
                num_changing_corrects += 1

            neg_phrase_changing_scores.append(one_pos_scores[0])
            phrase_changing_rankings = sorted(neg_phrase_changing_scores, reverse=True)
            rank = phrase_changing_rankings.index(one_pos_scores[0])
            ranking_changing.append(1. / (rank+1))

        if num_phrase_swapping != 0:
            neg_phrase_swapping_scores = one_neg_scores[num_phrase_changing : num_phrase_changing+num_phrase_swapping]
            num_swapping_examples += 1

            for j, score in enumerate(neg_phrase_swapping_scores):
                new_num_swapping_examples += 1
                if one_pos_scores[0] > score:
                    new_num_swapping_corrects += 1
                
            if one_pos_scores[0] > max(neg_phrase_swapping_scores):
                num_swapping_corrects += 1

            neg_phrase_swapping_scores.append(one_pos_scores[0])
            phrase_swapping_rankings = sorted(neg_phrase_swapping_scores, reverse=True)
            rank = phrase_swapping_rankings.index(one_pos_scores[0])
            ranking_swapping.append(1. / (rank+1))

        if num_phrase_dropping != 0:
            neg_phrase_dropping_scores = one_neg_scores[num_phrase_changing+num_phrase_swapping : num_phrase_changing+num_phrase_swapping+num_phrase_dropping]
            num_dropping_examples += 1
        
            for j, score in enumerate(neg_phrase_dropping_scores):
                new_num_dropping_examples += 1
                if one_pos_scores[0] > score:
                    new_num_dropping_corrects += 1

            if one_pos_scores[0] > max(neg_phrase_dropping_scores):
                num_dropping_corrects += 1

            neg_phrase_dropping_scores.append(one_pos_scores[0])
            phrase_dropping_rankings = sorted(neg_phrase_dropping_scores, reverse=True)
            rank = phrase_dropping_rankings.index(one_pos_scores[0])
            ranking_dropping.append(1. / (rank+1))


        if num_random != 0:
            neg_random_scores = one_neg_scores[-num_random: ]
            num_random_examples += 1

            for j, score in enumerate(neg_random_scores):
                new_num_random_examples += 1
                if one_pos_scores[0] > score:
                    new_num_random_corrects += 1
                
            if one_pos_scores[0] > max(neg_random_scores):
                num_random_corrects += 1

            neg_random_scores.append(one_pos_scores[0])
            random_rankings = sorted(neg_random_scores, reverse=True)
            rank = random_rankings.index(one_pos_scores[0])
            ranking_random.append(1. / (rank+1))

        # scores.append({"Index": itr_ind, "pos_score": one_pos_scores[0], "neg_socre": one_neg_scores, "margin": np.subtract(one_pos_scores, one_neg_scores).tolist()})
    # if not os.path.exists(f'analysis/{out_format}'):
    #     os.makedirs(f'analysis/{out_format}')
    # with open(f'analysis/{out_format}/{data_type}_validation_acc_epoch_{epoch}.json', 'w') as f:
    #     json.dump(scores, f, indent=4)
    total_acc = 1. * num_corrects / num_examples
    total_new_acc = 1. * new_num_corrects / new_num_examples
    total_mrr = ranking_sum / num_examples

    changing_acc = 1. * num_changing_corrects / num_changing_examples if num_changing_examples != 0 else 0
    changing_new_acc = 1. * new_num_changing_corrects / new_num_changing_examples if new_num_changing_examples != 0 else 0
    changing_mrr = sum(ranking_changing) / len(ranking_changing) if len(ranking_changing) != 0 else 0

    swapping_acc = 1. * num_swapping_corrects / num_swapping_examples if num_swapping_examples != 0 else 0
    swapping_new_acc = 1. * new_num_swapping_corrects / new_num_swapping_examples if new_num_swapping_examples != 0 else 0
    swapping_mrr = sum(ranking_swapping) / len(ranking_swapping) if len(ranking_swapping) != 0 else 0

    dropping_acc = 1. * num_dropping_corrects / num_dropping_examples if num_dropping_examples != 0 else 0
    dropping_new_acc = 1. * new_num_dropping_corrects / new_num_dropping_examples if new_num_dropping_examples != 0 else 0
    dropping_mrr = sum(ranking_dropping) / len(ranking_dropping) if len(ranking_dropping) != 0 else 0

    random_acc = 1. * num_random_corrects / num_random_examples if num_random_examples != 0 else 0
    random_new_acc = 1. * new_num_random_corrects / new_num_random_examples if new_num_random_examples != 0 else 0
    random_mrr = sum(ranking_random) / len(ranking_random) if len(ranking_random) != 0 else 0

    return (total_acc, total_new_acc, total_mrr, num_examples), \
           (changing_acc, changing_new_acc, changing_mrr, num_phrase_changing), \
           (swapping_acc, swapping_new_acc, swapping_mrr, num_phrase_swapping), \
           (dropping_acc, dropping_new_acc, dropping_mrr, num_phrase_dropping), \
           (random_acc, random_new_acc, random_mrr, num_random_examples)


def main():
    logger.info('********************  Spider Alignment  ********************')
    use_autoencoder = False

    # Directory of train/dev/test data files
    table_file = '../data/spider/tables.json'
    train_data_file = '../data/splash/train_10.json'
    dev_data_file = '../data/splash/dev_w_template_feedback.json'
    # test_data_file = '../data/splash/test_w_template_feedback.json'

    #Load training data
    train_align_dataset = SpiderAlignDataset(table_file=table_file, data_file=train_data_file, n_random=n_random, n_changing=n_changing, n_swapping=n_swapping, n_dropping=n_dropping, data_type='train',
                                             negative_sampling_mode=replace_mode, tokenizer_shortcut=model_shortcut)
    train_dataloader, _neg_len, _neg_tag = train_align_dataset.get_dataloader(batch_size=batch_size, shuffle=True, num_workers=4)
    dev_train_dataloader, train_neg_len, train_neg_tag = train_align_dataset.get_dataloader(batch_size=batch_size, shuffle=False, num_workers=4)


    # logger.info('****************** train_data *********************')
    # print("Train Dataloader Len: ", len(train_dataloader))
    # train_iter = iter(train_dataloader)  
    # (positive_tensors, negative_tensors), (positive_weight_matrix, negative_weight_matrix), (positive_lengths, negative_lengths), (positive_texts, negative_texts) = next(iter(train_iter))

    # print('***** Train positive *****\n')
    # print(positive_texts)
    # print('***** Train negative *****\n')
    # print(negative_texts)

    # Load dev all negative examples
    dev_all_dataset = SpiderAlignDataset(table_file=table_file, data_file=dev_data_file, n_random=n_random, n_changing=n_changing, n_swapping=n_swapping, n_dropping=n_dropping, data_type='dev',
                                           negative_sampling_mode=replace_mode, tokenizer_shortcut=model_shortcut)
    dev_all_dataloader, dev_neg_len, dev_neg_tag = dev_all_dataset.get_dataloader(batch_size=batch_size, shuffle=False, num_workers=4)
    # print("Dev Dataloader Len: ", len(dev_all_dataloader))

    
    # Training the aligner model
    out_format = f'{date}/{replace_mode}-{similarity_mode}-{pre}-lr-{lr}-bert-{blr}-m-{mar}-l1-{l1}-cls-{cls_weight}-pos-prior-{prior_pos}-neg-prior-{prior_neg}'

    aligner_model = BertAlignerModel(use_autoencoder=use_autoencoder, similarity_mode=similarity_mode, model_type=model_shortcut, bert_lr=blr, margin=mar, l1=l1, prior_pos=prior_pos, prior_neg=prior_neg)
    if os.path.exists(f'saved/{out_format}/model.pt'):
        aligner_model.load_state_dict(torch.load(f'saved/{out_format}/model.pt'))
    if torch.cuda.is_available():
        aligner_model = aligner_model.cuda()
    else:
        logger.warning("Model is running on CPU. The progress will be very slow.")

    # aligner_model.load_state_dict(torch.load('saved/splash/model-82-mix-recall.pt', map_location='cpu'))
    criterion = aligner_model.criterion
    optimizer = aligner_model.optimizer

    early_stop_count = 0
    last_val_dev_loss = sys.float_info.max

    for epoch in range(epochs):
    #     """
    #     aligner_model = BertAlignerModel(use_autoencoder=use_autoencoder, similarity_mode=similarity_mode)
    #     aligner_model.load_state_dict(torch.load(f'saved/mix-50-100-lr-1e5-bert-1e7-l1-1e1-m-0.6/model-{epoch+1}.pt'))
    #     if torch.cuda.is_available():
    #         aligner_model = aligner_model.cuda()
    #     else:
    #         logger.warning("Model is running on CPU. The progress will be very slow.")
    #     criterion = aligner_model.criterion
    #     """
        if early_stop_count >= 5:
            print("Early Stopping!")
            break       

        train_loss, train_pure_loss, train_l1_term, train_pos_prior, train_neg_prior, bat_train_loss, bat_train_pure_loss, bat_train_l1_term \
            = train(aligner_model, train_dataloader, criterion, optimizer)
        val_train_loss, val_train_pure_loss, val_train_l1_term, val_train_pos_prior, val_train_neg_prior, val_train_acc_all, val_train_acc_changing, val_train_acc_swapping, val_train_acc_dropping, val_train_acc_random \
            = validate(aligner_model, dev_train_dataloader, criterion, train_neg_len, train_neg_tag, epoch, "train", out_format, similarity_mode)

        val_train_acc, val_train_acc_new, val_train_acc_mrr, val_train_total_num_neg = val_train_acc_all
        val_train_changing_acc, val_train_changing_acc_new, val_train_changing_acc_mrr, val_train_num_changing = val_train_acc_changing
        val_train_swapping_acc, val_train_swapping_acc_new, val_train_swapping_acc_mrr, val_train_num_swapping = val_train_acc_swapping
        val_train_dropping_acc, val_train_dropping_acc_new, val_train_dropping_acc_mrr, val_train_num_dropping = val_train_acc_dropping
        val_train_random_acc, val_train_random_acc_new, val_train_random_acc_mrr, val_train_num_random = val_train_acc_random

        # print(f"Train loss: {train_loss}, Train Accuracy: Original: {val_train_acc}, Revised: {val_train_acc_new}, MRR: {val_train_acc_mrr}, Count: {val_train_total_num_neg}")
        # print(f"Train Changing Accuracy: Original: {val_train_changing_acc}, Revised: {val_train_changing_acc_new}, MRR: {val_train_changing_acc_mrr}, Count: {val_train_num_changing}")
        # print(f"Train Swapping Accuracy: Original: {val_train_swapping_acc}, Revised: {val_train_swapping_acc_new}, MRR: {val_train_swapping_acc_mrr}, Count: {val_train_num_swapping}")
        # print(f"Train Dropping Accuracy: Original: {val_train_dropping_acc}, Revised: {val_train_dropping_acc_new}, MRR: {val_train_dropping_acc_mrr}, Count: {val_train_num_dropping}")
        # print(f"Train Random Accuracy: Original: {val_train_random_acc}, Revised: {val_train_random_acc_new}, MRR: {val_train_random_acc_mrr}, Count: {val_train_num_random}")

        val_dev_loss, val_dev_pure_loss, val_dev_l1_term, val_dev_pos_prior, val_dev_neg_prior, val_dev_acc_all, val_dev_acc_changing, val_dev_acc_swapping, val_dev_acc_dropping, val_dev_acc_random \
            = validate(aligner_model, dev_all_dataloader, criterion, dev_neg_len, dev_neg_tag, epoch, "dev", out_format, similarity_mode)
        val_dev_acc, val_dev_acc_new, val_dev_acc_mrr, val_dev_total_num_neg = val_dev_acc_all
        val_dev_changing_acc, val_dev_changing_acc_new, val_dev_changing_acc_mrr, val_dev_num_changing = val_dev_acc_changing
        val_dev_swapping_acc, val_dev_swapping_acc_new, val_dev_swapping_acc_mrr, val_dev_num_swapping = val_dev_acc_swapping
        val_dev_dropping_acc, val_dev_dropping_acc_new, val_dev_dropping_acc_mrr, val_dev_num_dropping = val_dev_acc_dropping
        val_dev_random_acc, val_dev_random_acc_new, val_dev_random_acc_mrr, val_dev_num_random = val_dev_acc_random

        # print(f"Dev loss: {val_dev_loss}, dev Accuracy: Original: {val_dev_acc}, Revised: {val_dev_acc_new}, MRR: {val_dev_acc_mrr}, Count: {val_dev_total_num_neg}")
        # print(f"Dev Changing Accuracy: Original: {val_dev_changing_acc}, Revised: {val_dev_changing_acc_new}, MRR: {val_dev_changing_acc_mrr}, Count: {val_dev_num_changing}")
        # print(f"Dev Swapping Accuracy: Original: {val_dev_swapping_acc}, Revised: {val_dev_swapping_acc_new}, MRR: {val_dev_swapping_acc_mrr}, Count: {val_dev_num_swapping}")
        # print(f"Dev Dropping Accuracy: Original: {val_dev_dropping_acc}, Revised: {val_dev_dropping_acc_new}, MRR: {val_dev_dropping_acc_mrr}, Count: {val_dev_num_dropping}")
        # print(f"Dev Random Accuracy: Original: {val_dev_random_acc}, Revised: {val_dev_random_acc_new}, MRR: {val_dev_random_acc_mrr}, Count: {val_dev_num_random}")

        if val_dev_loss > last_val_dev_loss:
            early_stop_count += 1
        else:
            early_stop_count = 0

        last_val_dev_loss = val_dev_loss


        wandb.log({"train_loss": train_loss, "val_train_loss": val_train_loss, "val_dev_loss": val_dev_loss, \
            "val_train_pure_loss": val_train_pure_loss, "val_train_l1_term": val_train_l1_term, "val_train_pos_prior": val_train_pos_prior, "val_train_neg_prior": val_train_neg_prior, \
            "val_dev_pure_loss": val_dev_pure_loss, "val_dev_l1_term": val_dev_l1_term, "val_dev_pos_prior": val_dev_pos_prior, "val_dev_neg_prior": val_dev_neg_prior, \
            "train_acc": val_train_acc, "train_acc_new": val_train_acc_new, "train_acc_mrr": val_train_acc_mrr, \
            "train_changing_acc": val_train_changing_acc, "train_swapping_acc": val_train_swapping_acc, "train_dropping_acc": val_train_dropping_acc, "train_random_acc": val_train_random_acc, \
            "train_changing_acc_new": val_train_changing_acc_new, "train_swapping_acc_new": val_train_swapping_acc_new, "train_dropping_acc_new": val_train_dropping_acc_new, "train_random_acc_new": val_train_random_acc_new, \
            "train_changing_acc_mrr": val_train_changing_acc_mrr, "train_swapping_acc_mrr": val_train_swapping_acc_mrr, "train_dropping_acc_mrr": val_train_dropping_acc_mrr, "train_random_acc_mrr": val_train_random_acc_mrr, \
            "dev_acc": val_dev_acc, "dev_acc_new": val_dev_acc_new, "dev_acc_mrr": val_dev_acc_mrr, \
            "dev_changing_acc": val_dev_changing_acc, "dev_swapping_acc": val_dev_swapping_acc, "dev_dropping_acc": val_dev_dropping_acc, "dev_random_acc": val_dev_random_acc, \
            "dev_changing_acc_new": val_dev_changing_acc_new, "dev_swapping_acc_new": val_dev_swapping_acc_new, "dev_dropping_acc_new": val_dev_dropping_acc_new, "dev_random_acc_new": val_dev_random_acc_new, \
            "dev_changing_acc_mrr": val_dev_changing_acc_mrr, "dev_swapping_acc_mrr": val_dev_swapping_acc_mrr, "dev_dropping_acc_mrr": val_dev_dropping_acc_mrr, "dev_random_acc_mrr": val_dev_random_acc_mrr})


        if not os.path.exists(f'./saved/{out_format}'):
            os.makedirs(f'./saved/{out_format}')
        torch.save(aligner_model.state_dict(), f'saved/{out_format}/model-{epoch}.pt')

if __name__ == '__main__':
    main()