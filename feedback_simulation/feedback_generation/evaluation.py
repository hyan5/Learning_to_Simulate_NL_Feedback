from aligner_model import BertAligner, BertAlignerModel
import numpy as np
import json
import torch
from scipy.optimize import linear_sum_assignment
import pdb

from nltk import word_tokenize

class EvalMetric:
  def __init__(self, span_weight=0.9, model_type='roberta-large', checkpoints='eval_ckp/evaluation.pt'):
    self.span_weight = span_weight
    self.model_type = model_type
    self.checkpoints = checkpoints

  def calculate_similarity_matrix(self, preds, refs, span_list):
    bert_model = BertAlignerModel(use_autoencoder=False, model_type=self.model_type)
    bert_model.load_state_dict(torch.load(self.checkpoints))
    bert_model.eval()
    model = BertAligner(aligner_model=bert_model)

    res_pre =[]
    res_rec =[]
    res_f1 =[]
    res_avg =[]

    for can, ref, spans in zip(preds, refs, span_list):
        can = ' '.join(word_tokenize(can)).replace("``", "\"").replace("''", "\"")
        alignment_matrix, cls_similarity, temp, pos, primary_span, secondary_span = model.calculate_alignment(ref, can, spans)

        # Template and candidate feedback
        tokens1, tokens2 = temp[1:-1], pos[1:-1]
        # cls_similarity = cls_similarity.item()
        alignment_matrix = alignment_matrix.squeeze(0).detach().cpu().numpy()

        # Calculate precision and recall matrix  
        precision_matrix, recall_matrix = alignment_matrix.max(0), alignment_matrix.max(-1)

        # Calculate unweighted score
        precision_score, recall_score = np.sum(precision_matrix) / (1. * len(tokens2)), np.sum(recall_matrix) / (1. * len(tokens1))
        average_score = (precision_score + recall_score) / 2
        f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
        
        # Extract all secondary spans
        all_secondary = []
        current_start = 0
        for start, end in primary_span:
            if current_start < start:
                all_secondary.append((current_start, start-1))
            current_start = end + 1
        if current_start - 1 != len(tokens1):
            all_secondary.append((current_start, len(tokens1)-1))

        # Weight matrix
        weight_mat = np.zeros(alignment_matrix.shape[0])

        # Get the row index of the maximum values for each column
        precision_row_ind = np.argmax(alignment_matrix, axis=0)

        # Primary Span weight
        for start, end in primary_span:
            weight_mat[start : end+1] = self.span_weight

        # Secondary Span weight
        for start, end in all_secondary:
            weight_mat[start : end+1] = 1 - self.span_weight

        # Calculate the weight matrix for column
        col_weight_mat = np.array([weight_mat[row] for row in precision_row_ind])

        # Bipartite Matching
        row_ind, col_ind = linear_sum_assignment(-alignment_matrix)

        # Calculate the primary score and secondary score in Bipartite Matching
        bi_primary_score = 0
        bi_secondary_score = 0
        bi_pre_primary_score = 0
        bi_pre_secondary_score = 0

        # Count the number of primary and secondary tokens respectively
        bi_primary_tokens = 0
        bi_secondary_tokens = 0
        bi_pre_primary_tokens = 0
        bi_pre_secondary_tokens = 0

        for i in range(len(tokens1)):
            if i not in row_ind:
                if weight_mat[i] == self.span_weight:
                    bi_primary_score += recall_matrix[i]
                    bi_primary_tokens +=1
                else:
                    bi_secondary_score += recall_matrix[i]
                    bi_secondary_tokens += 1
        
        for i in range(len(tokens2)):
            if i not in col_ind:
                if col_weight_mat[i] == self.span_weight:
                    bi_pre_primary_score += precision_matrix[i]
                    bi_pre_primary_tokens += 1
                else:
                    bi_pre_secondary_score += precision_matrix[i]
                    bi_pre_secondary_tokens += 1

        for row, col in zip(row_ind, col_ind):
            if weight_mat[row] == self.span_weight:
                bi_primary_score += alignment_matrix[row][col]
                bi_primary_tokens += 1
            else:
                bi_secondary_score += alignment_matrix[row][col]
                bi_secondary_tokens += 1
            
            if col_weight_mat[col] == self.span_weight:
                bi_pre_primary_score += alignment_matrix[row][col]
                bi_pre_primary_tokens += 1
            else:
                bi_pre_secondary_score += alignment_matrix[row][col]
                bi_pre_secondary_tokens += 1

        # Calculate the unweighted Bipartite score
        # bi_precision = (bi_pre_primary_score + bi_pre_secondary_score) / len(tokens2)
        # bi_recall = (bi_primary_score + bi_secondary_score) / len(tokens1)
        # bi_f1 = 2 * bi_precision * bi_recall / (bi_precision + bi_recall)
        # bi_avg = (bi_precision + bi_recall) / 2

        # Calculate the weighted score
        # weighted_recall = recall_matrix@weight_mat / np.sum(weight_mat)
        # weighted_precision = precision_matrix@col_weight_mat / np.sum(col_weight_mat)
        # weighted_f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
        # weighted_avg = (weighted_precision + weighted_recall) / 2
        
        # Calculate the weighted Bipartite score
        weighted_bi_recall = (bi_primary_score * self.span_weight + bi_secondary_score * (1 - self.span_weight)) / (bi_primary_tokens * self.span_weight + bi_secondary_tokens * (1 - self.span_weight))
        weighted_bi_precision = (bi_pre_primary_score * self.span_weight + bi_pre_secondary_score * (1 - self.span_weight)) / (bi_pre_primary_tokens * self.span_weight + bi_pre_secondary_tokens * (1 - self.span_weight))
        weighted_bi_f1 = 2 * weighted_bi_recall * weighted_bi_precision / (weighted_bi_precision + weighted_bi_recall)
        weighted_bi_avg = (weighted_bi_precision + weighted_bi_recall) / 2

        res_pre.append(weighted_bi_recall)
        res_rec.append(weighted_bi_precision)
        res_f1.append(weighted_bi_f1)
        res_avg.append(weighted_bi_avg)

    results = {'weighted_bipartite_averge': np.mean(np.array(res_avg)), 'weighted_bipartite_f1': np.mean(np.array(res_f1))}
    return results

def load_json(file_path):
    return json.load(open(file_path, 'r'))

def read_file(file_path):
    results = []
    with open(file_path, 'r') as f:
        results = [line.strip() for line in f.readlines()]
    return results

if __name__ == '__main__':
    dev = load_json('../../data/splash/dev_w_template_feedback.json')
    sim = read_file('sim_data.sim')

    temp = [data['template_feedback'] for data in dev]
    pri = [data['primary_span'] for data in dev]
    sec = [data['secondary_span'] for data in dev]
    metric = EvalMetric()
    score = metric.calculate_similarity_matrix(sim, temp, zip(pri, sec))