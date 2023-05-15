from asyncore import write
import enum
from aligner_model import BertAligner, BertAlignerModel
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import json
import torch
from scipy.optimize import linear_sum_assignment
import dill
import pdb
import bert_score
import random
from nltk import word_tokenize

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

smooth_fc = SmoothingFunction()

def calculate_similarity_matrix(model, ref, can, spans=None, span_weight=0, showimg=False):
    # print(f'Ref: {ref}')
    # print(f'Can: {can}')

    # Calculate BLEU
    bleu_score = sentence_bleu([word_tokenize(ref)], word_tokenize(can), smoothing_function=smooth_fc.method1)

    # Calculate BERTScore
    P, R, F = bert_score.score([can], [ref], lang='en')
    # bert_score.plot_example(ref, can, lang='en', fname='out.png')

    alignment_matrix, cls_similarity, temp, pos, primary_span, secondary_span = model.calculate_alignment(ref, can, spans)

    # Template and candidate feedback
    tokens1, tokens2 = temp[1:-1], pos[1:-1]
    cls_similarity = cls_similarity.item()
    alignment_matrix = alignment_matrix.squeeze(0).detach().cpu().numpy()

    # Calculate precision and recall matrix  
    precision_matrix, recall_matrix = alignment_matrix.max(0), alignment_matrix.max(-1)

    # Calculate unweighted score
    precision_score, recall_score = np.sum(precision_matrix) / (1. * len(tokens2)), np.sum(recall_matrix) / (1. * len(tokens1))
    average_score = (precision_score + recall_score) / 2
    f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)

    # final_score = (1 - cls_weight) * average_score + cls_weight * cls_sim
    # final_score_ori = (1 - cls_weight) * average_score + cls_weight * cls_similarity
    
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
        weight_mat[start : end+1] = span_weight

    # Secondary Span weight
    for start, end in all_secondary:
        weight_mat[start : end+1] = 1 - span_weight

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
            if weight_mat[i] == span_weight:
                bi_primary_score += recall_matrix[i]
                bi_primary_tokens +=1
            else:
                bi_secondary_score += recall_matrix[i]
                bi_secondary_tokens += 1
    
    for i in range(len(tokens2)):
        if i not in col_ind:
            if col_weight_mat[i] == span_weight:
                bi_pre_primary_score += precision_matrix[i]
                bi_pre_primary_tokens += 1
            else:
                bi_pre_secondary_score += precision_matrix[i]
                bi_pre_secondary_tokens += 1

    for row, col in zip(row_ind, col_ind):
        if weight_mat[row] == span_weight:
            bi_primary_score += alignment_matrix[row][col]
            bi_primary_tokens += 1
        else:
            bi_secondary_score += alignment_matrix[row][col]
            bi_secondary_tokens += 1
        
        if col_weight_mat[col] == span_weight:
            bi_pre_primary_score += alignment_matrix[row][col]
            bi_pre_primary_tokens += 1
        else:
            bi_pre_secondary_score += alignment_matrix[row][col]
            bi_pre_secondary_tokens += 1

    # Calculate the unweighted Bipartite score
    bi_precision = (bi_pre_primary_score + bi_pre_secondary_score) / len(tokens2)
    bi_recall = (bi_primary_score + bi_secondary_score) / len(tokens1)
    bi_f1 = 2 * bi_precision * bi_recall / (bi_precision + bi_recall)
    bi_avg = (bi_precision + bi_recall) / 2

    # print(alignment_matrix.shape) 


    # Calculate the weighted score
    weighted_recall = recall_matrix@weight_mat / np.sum(weight_mat)
    weighted_precision = precision_matrix@col_weight_mat / np.sum(col_weight_mat)
    weighted_f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
    weighted_avg = (weighted_precision + weighted_recall) / 2
  
    # Calculate the weighted Bipartite score
    weighted_bi_recall = (bi_primary_score * span_weight + bi_secondary_score * (1 - span_weight)) / (bi_primary_tokens * span_weight + bi_secondary_tokens * (1 - span_weight))
    weighted_bi_precision = (bi_pre_primary_score * span_weight + bi_pre_secondary_score * (1 - span_weight)) / (bi_pre_primary_tokens * span_weight + bi_pre_secondary_tokens * (1 - span_weight))
    weighted_bi_f1 = 2 * weighted_bi_recall * weighted_bi_precision / (weighted_bi_precision + weighted_bi_recall)
    weighted_bi_avg = (weighted_bi_precision + weighted_bi_recall) / 2
    
    # print(f'BERTScore: P: {P}, R: {R}, F1: {F}, Avg: {(P+R)/2}')

    # print(f'Bipartite Matching Row: {row_ind}')
    # print(f'Bipartite Matching Col: {col_ind}')


    # print(f'P: {precision_score}, R: {recall_score}, F1: {f1_score}, Avg: {average_score}')
    # print(f'Bipartite Score: P: {bi_precision}, R: {bi_recall}, F1: {bi_f1}, Avg: {bi_avg}')

    # print(f'Weighted Score: P: {weighted_precision}, R: {weighted_recall}, F1: {weighted_f1}, Avg: {weighted_avg}')
    # print(f'Weighted Bipartite Score: P: {weighted_bi_precision}, R: {weighted_bi_recall}, F1: {weighted_bi_f1}, Avg: {weighted_bi_avg}')
    
    if showimg:
        fig, ax = plt.subplots(figsize=(len(tokens1)*1.5, len(tokens2)*1.5))
        im = ax.imshow(alignment_matrix, cmap="Blues", vmin=0, vmax=1)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(tokens2)))
        ax.set_yticks(np.arange(len(tokens1)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(tokens2, fontsize=10)
        ax.set_yticklabels(tokens1, fontsize=10)
        ax.grid(False)
        plt.xlabel("Candidate (tokenized)", fontsize=14)
        plt.ylabel("Reference (tokenized)", fontsize=14)
        plt.title(f"Similarity Matrix", fontsize=14)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)
        fig.colorbar(im, cax=cax)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(len(tokens1)):
            for j in range(len(tokens2)):
                text = ax.text(
                    j,
                    i,
                    "{:.3f}".format(alignment_matrix[i, j]),
                    ha="center",
                    va="center",
                    color="k" if alignment_matrix[i, j] < 0.5 else "w",
                )
        fig.tight_layout()
        plt.savefig(f'avg.png')
        plt.show()

    return weighted_bi_avg, F.item(), bleu_score

def draw_hist(values, labels, title):
    x = np.arange(len(labels))

    fig, ax = plt.subplots()
    rect = ax.bar(x, values, 0.2)
    ax.set_ylabel('Margin')
    ax.set_xlabel('Primary Span Weight')

    ax.set_title(f'The Margin on Different Weights ({title})')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rect, padding=1)

    fig.tight_layout()
    plt.show()

def load_json(file_dir):
    return json.load(open(file_dir, 'r'))

def load_file(file_dir):
    results = []
    with open(file_dir, 'r') as f:
        results = [line for line in f.readlines()]
    return results

if  __name__ == '__main__':
    span_weight = 0.9
    showimg = False
    model_type = 'roberta-large'

    bert_model = BertAlignerModel(use_autoencoder=False, model_type=model_type)
    bert_model.load_state_dict(torch.load(f'saved/model-99.pt'))
    bert_model.eval()
    bert_aligner = BertAligner(aligner_model=bert_model)

    
    dev_data = load_json('feedback_eval/dev_w_edits.json')
    sim_bart_cwqes = load_file('feedback_eval/cwqes/dev.sim')
    sim_bart_dqes = load_file('feedback_eval/dqes/dev.sim')
    sim_bart_tqes = load_file('feedback_eval/tqes/dev.sim')

    sim_t5_cwqes = load_file('feedback_eval/cwqes/cwqes_dev.sim')
    sim_t5_dqes = load_file('feedback_eval/dqes/dqes_dev.sim')
    sim_t5_tqes = load_file('feedback_eval/tqes/tqes_dev.sim')

    random.seed(41)
    ran_list = random.sample(range(len(dev_data)), 50)
    print(f'Random List: {len(ran_list)}')

    bart_bleu_c = []
    bart_bert_c = []
    bart_eval_c = []

    bart_bleu_d = []
    bart_bert_d = []
    bart_eval_d = []

    bart_bleu_t = []
    bart_bert_t = []
    bart_eval_t = []

    t5_bleu_c = []
    t5_bert_c = []
    t5_eval_c = []

    t5_bleu_d = []
    t5_bert_d = []
    t5_eval_d = []

    t5_bleu_t = []
    t5_bert_t = []
    t5_eval_t = []

    out_json = []
    for idx in ran_list:
        out_exp = {}
        _q = dev_data[idx]['question']
        _c = dev_data[idx]['gold_parse']
        _w = dev_data[idx]['predicted_parse_with_values']
        _d = dev_data[idx]['edits_original']
        _e = dev_data[idx]['predicted_parse_explanation']
        _t = dev_data[idx]['template_feedback']
        fed = dev_data[idx]['feedback']
        pri_span = dev_data[idx]['primary_span']
        sec_span = dev_data[idx]['secondary_span']

        bart_c = sim_bart_cwqes[idx]
        bart_d = sim_bart_dqes[idx]
        bart_t = sim_bart_tqes[idx]

        t5_c = sim_t5_cwqes[idx]
        t5_d = sim_t5_dqes[idx]
        t5_t = sim_t5_tqes[idx]

        bart_c_eval, bart_c_bert, bart_c_bleu = calculate_similarity_matrix(bert_aligner, _t, bart_c, spans=(pri_span, sec_span), span_weight=span_weight)
        bart_d_eval, bart_d_bert, bart_d_bleu = calculate_similarity_matrix(bert_aligner, _t, bart_d, spans=(pri_span, sec_span), span_weight=span_weight)
        bart_t_eval, bart_t_bert, bart_t_bleu = calculate_similarity_matrix(bert_aligner, _t, bart_t, spans=(pri_span, sec_span), span_weight=span_weight)

        t5_c_eval, t5_c_bert, t5_c_bleu = calculate_similarity_matrix(bert_aligner, _t, t5_c, spans=(pri_span, sec_span), span_weight=span_weight)
        t5_d_eval, t5_d_bert, t5_d_bleu = calculate_similarity_matrix(bert_aligner, _t, t5_d, spans=(pri_span, sec_span), span_weight=span_weight)
        t5_t_eval, t5_t_bert, t5_t_bleu = calculate_similarity_matrix(bert_aligner, _t, t5_t, spans=(pri_span, sec_span), span_weight=span_weight)

        bart_bleu_c.append(bart_c_bleu)
        bart_bleu_d.append(bart_d_bleu)
        bart_bleu_t.append(bart_t_bleu)

        bart_bert_c.append(bart_c_bert)
        bart_bert_d.append(bart_d_bert)
        bart_bert_t.append(bart_t_bert)

        bart_eval_c.append(bart_c_eval)
        bart_eval_d.append(bart_d_eval)
        bart_eval_t.append(bart_t_eval)

        t5_bleu_c.append(t5_c_bleu)
        t5_bleu_d.append(t5_d_bleu)
        t5_bleu_t.append(t5_t_bleu)

        t5_bert_c.append(t5_c_bert)
        t5_bert_d.append(t5_d_bert)
        t5_bert_t.append(t5_t_bert)

        t5_eval_c.append(t5_c_eval)
        t5_eval_d.append(t5_d_eval)
        t5_eval_t.append(t5_t_eval)

        out_exp['question'] = _q
        out_exp['gold_parse'] = _c
        out_exp['predicted_parse'] = _w
        out_exp['feedback'] = fed
        out_exp['template'] = _t
        out_exp['explanation'] = _e
        out_exp['edits'] = _d

        out_exp['BART_SIM'] = [{'cwqes': bart_c, 'BLEU': bart_c_bleu, 'BERT': bart_c_bert, 'EVAL': bart_c_eval}, \
                               {'dqes': bart_d, 'BLEU': bart_d_bleu, 'BERT': bart_d_bert, 'EVAL': bart_d_eval}, \
                               {'tqes': bart_t, 'BLEU': bart_t_bleu, 'BERT': bart_t_bert, 'EVAL': bart_t_eval}]

        out_exp['T5_SIM'] = [{'cwqes': t5_c, 'BLEU': t5_c_bleu, 'BERT': t5_c_bert, 'EVAL': t5_c_eval}, \
                               {'dqes': t5_d, 'BLEU': t5_d_bleu, 'BERT': t5_d_bert, 'EVAL': t5_d_eval}, \
                               {'tqes': t5_t, 'BLEU': t5_t_bleu, 'BERT': t5_t_bert, 'EVAL': t5_t_eval}]

        out_json.append(out_exp)

    bart_avg_bleu_c = np.mean(np.array(bart_bleu_c))
    bart_avg_bert_c = np.mean(np.array(bart_bert_c))
    bart_avg_eval_c = np.mean(np.array(bart_eval_c))

    bart_avg_bleu_d = np.mean(np.array(bart_bleu_d))
    bart_avg_bert_d = np.mean(np.array(bart_bert_d))
    bart_avg_eval_d = np.mean(np.array(bart_eval_d))

    bart_avg_bleu_t = np.mean(np.array(bart_bleu_t))
    bart_avg_bert_t = np.mean(np.array(bart_bert_t))
    bart_avg_eval_t = np.mean(np.array(bart_eval_t))

    t5_avg_bleu_c = np.mean(np.array(t5_bleu_c))
    t5_avg_bert_c = np.mean(np.array(t5_bert_c))
    t5_avg_eval_c = np.mean(np.array(t5_eval_c))

    t5_avg_bleu_d = np.mean(np.array(t5_bleu_d))
    t5_avg_bert_d = np.mean(np.array(t5_bert_d))
    t5_avg_eval_d = np.mean(np.array(t5_eval_d))


    t5_avg_bleu_t = np.mean(np.array(t5_bleu_t))
    t5_avg_bert_t = np.mean(np.array(t5_bert_t))
    t5_avg_eval_t = np.mean(np.array(t5_eval_t))


    out_json.append({'BART_BLEU_AVG_CWQES': bart_avg_bleu_c, 'BART_BLEU_AVG_DQES': bart_avg_bleu_d, 'BART_BLEU_AVG_TQES': bart_avg_bleu_t})
    out_json.append({'BART_BERT_AVG_CWQES': bart_avg_bert_c, 'BART_BERT_AVG_DQES': bart_avg_bert_d, 'BART_BERT_AVG_TQES': bart_avg_bert_t})
    out_json.append({'BART_EVAL_AVG_CWQES': bart_avg_eval_c, 'BART_EVAL_AVG_DQES': bart_avg_eval_d, 'BART_EVAL_AVG_TQES': bart_avg_eval_t})
    out_json.append({'T5_BLEU_AVG_CWQES': t5_avg_bleu_c, 'T5_BLEU_AVG_DQES': t5_avg_bleu_d, 'T5_BLEU_AVG_TQES': t5_avg_bleu_t})
    out_json.append({'T5_BERT_AVG_CWQES': t5_avg_bert_c, 'T5_BERT_AVG_DQES': t5_avg_bert_d, 'T5_BERT_AVG_TQES': t5_avg_bert_t})
    out_json.append({'T5_EVAL_AVG_CWQES': t5_avg_eval_c, 'T5_EVAL_AVG_DQES': t5_avg_eval_d, 'T5_EVAL_AVG_TQES': t5_avg_eval_t})
    
    with open(f'analysis_bart_t5.json') as f:
        json.dump(out_json, indent=4)
    
    # with open(f'analysis/mar_pd_{span_weight}.json') as f:
    #     json.dump(pd_margin, indent=4)

    # with open(f'analysis/mar_ran_{span_weight}.json') as f:
    #     json.dump(ran_margin, indent=4)