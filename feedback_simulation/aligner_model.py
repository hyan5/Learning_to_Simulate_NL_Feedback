# coding: utf-8

import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel

from data import BertUtil
from loss import HingeLoss
import pdb


class BatchCosineSimilarity(nn.Module):
    def __init__(self):
        super(BatchCosineSimilarity, self).__init__()
        self.cos = nn.CosineSimilarity(eps=1e-8)


    def forward(self, input1, input2):
        """
        input1 & input2 are both embedding sequences from rnn encoder
        """
        batch_dot_product = torch.bmm(input1, input2.transpose(1, 2))
        norm_1, norm_2 = torch.norm(input1, p=2, dim=-1), torch.norm(input2, p=2, dim=-1)
        norm_matrix = torch.bmm(torch.unsqueeze(norm_1, -1), torch.unsqueeze(norm_2, 1)) + 1e-8
        assert norm_matrix.size() == batch_dot_product.size()
        cosine_similarity = torch.div(batch_dot_product, norm_matrix)
        return cosine_similarity     

class BertAlignerModel(torch.nn.Module):
    def __init__(self, hidden_dim=100, use_autoencoder=False, similarity_mode='average', model_type='roberta-large', bert_lr=1e-8, margin=0.6, l1=0.1, prior_pos=1e-3, prior_neg=1e-3):
        self.use_autoencoder = use_autoencoder
        self.similarity_mode = similarity_mode
        self.model_type = model_type
        super().__init__()
        bert_pretrained_weights_shortcut = 'roberta-large'
        # bert_output_dim = 768
        self.hidden_dim = 1024

        self.bert_chosen_layer = -1
        
        self.bert_model = AutoModel.from_pretrained(model_type, output_hidden_states=True)
            
        self.dropout = nn.Dropout(p=0.5)
        
        self.similarity_layer = BatchCosineSimilarity()
        self.criterion = HingeLoss(margin=margin, l1_norm_weight=l1, pos_prior_weight=prior_pos, neg_prior_weight=prior_neg, similarity_mode=similarity_mode)
        self.optimizer = optim.Adam([
            {'params': self.bert_model.parameters(), 'lr': bert_lr},
        ])

    def forward(self, template_tensor, template_lengths, template_weight_matrix=None, template_att_matrix=None,
                positive_tensor=None, positive_lengths=None, positive_weight_matrix=None, positive_att_matrix=None,
                negative_tensor=None, negative_lengths=None, negative_weight_matrix=None, negative_att_matrix=None,
                ques_max_len=None, pos_max_len=None, neg_max_len=None, mode='train'):
        assert mode in ('train', 'eval')

        # Bert Encoding output
        template_bert_output_all = self.bert_model(template_tensor, attention_mask=template_att_matrix)
        if self.bert_chosen_layer == -1:
            template_bert_output = template_bert_output_all[0][:, 1:-1]
            template_cls = template_bert_output_all[0][:, 0]
        else:
            # All hidden states
            template_bert_hidden_all = template_bert_output_all[2]

            # Chose the second to last hidden layer
            template_bert_output = template_bert_hidden_all[self.bert_chosen_layer][:, 1:-1]
            template_cls = template_bert_hidden_all[self.bert_chosen_layer][:, 0]



        positive_bert_output_all = self.bert_model(positive_tensor, attention_mask=positive_att_matrix) # encoding
        
        if self.bert_chosen_layer == -1:
            positive_bert_output = positive_bert_output_all[0][:, 1:-1]
            positive_cls = positive_bert_output_all[0][:, 0]

        else:
            # All hidden states
            positive_bert_hidden_all = positive_bert_output_all[2]

            # Chose the second to last hidden layer
            positive_bert_output = positive_bert_hidden_all[self.bert_chosen_layer][:, 1:-1] 
            positive_cls = positive_bert_hidden_all[self.bert_chosen_layer][:, 0]

        batch_size = template_bert_output.size(0)
        
        # Max length of template feedback in the batch
        template_max_len = ques_max_len[0] if ques_max_len is not None else template_lengths.max()

        # Max length of human feedback in the batch
        positive_max_len = pos_max_len[0] if pos_max_len is not None else positive_lengths.max()

        # Tensor type
        _DeviceTensor = torch.cuda.FloatTensor if next(self.parameters()).is_cuda else torch.FloatTensor

        # Template feedback matrix
        # template_matrix = _DeviceTensor(batch_size, template_max_len, self.hidden_dim).zero_()

        # Template feedback bert output matrix
        template_matrix = _DeviceTensor(batch_size, template_max_len, self.hidden_dim).zero_()

        # Human feedback matrix
        # positive_matrix = _DeviceTensor(batch_size, positive_max_len, self.hidden_dim).zero_()

        # Human feedback bert output
        positive_matrix = _DeviceTensor(batch_size, positive_max_len, self.hidden_dim).zero_()

        # pdb.set_trace()

        if negative_tensor is not None:
            # negative_lengths = negative_lengths.permute(1, 0)
            negative_bert_output_all = self.bert_model(negative_tensor, attention_mask=negative_att_matrix)

            if self.bert_chosen_layer == -1:
                negative_bert_output = negative_bert_output_all[0][:, 1:-1]
                negative_cls = negative_bert_output_all[0][:, 0]
            else:
                negative_bert_hidden_all = negative_bert_output_all[2]

                negative_bert_output = negative_bert_hidden_all[self.bert_chosen_layer][:, 1:-1]
                negative_cls = negative_bert_hidden_all[self.bert_chosen_layer][:, 0]

            negative_max_len = neg_max_len[0] if neg_max_len is not None else negative_lengths.max()
            # negative_matrix = _DeviceTensor(batch_size, negative_max_len, self.hidden_dim).zero_()
            negative_matrix = _DeviceTensor(batch_size, negative_max_len, self.hidden_dim).zero_()

        for batch_idx in range(batch_size):
            template_matrix[batch_idx, :template_lengths[batch_idx]] = template_bert_output[batch_idx, :template_lengths[batch_idx]]
            positive_matrix[batch_idx, :positive_lengths[batch_idx]] = positive_bert_output[batch_idx, :positive_lengths[batch_idx]]

            if negative_tensor is not None:
                negative_matrix[batch_idx, :negative_lengths[batch_idx]] = negative_bert_output[batch_idx, :negative_lengths[batch_idx]]
                
        if mode == 'train':
            positive_similarity_matrix = self.similarity_layer(template_matrix, positive_matrix)

        else:
            positive_similarity_matrix = self.similarity_layer(template_matrix, positive_matrix)

        if negative_tensor is not None:
            if mode == 'train':
                negative_similarity_matrix = self.similarity_layer(template_matrix, negative_matrix)

            else:
                negative_similarity_matrix = self.similarity_layer(template_matrix, negative_matrix)

        if negative_tensor is not None:
            return positive_similarity_matrix, negative_similarity_matrix

        else:
            return positive_similarity_matrix


class BertAligner:
    def __init__(self, aligner_model = None):
        if aligner_model is not None:
            self.aligner_model = aligner_model
            print("aligner_model is provided")
        else:
            self.aligner_model = BertAlignerModel()
            if os.path.exists('saved/splash/model.pt'):
                self.aligner_model.load_state_dict(torch.load('saved/splash/model.pt'))
            else:
                logging.warning("No pretrined aligned model loaded!!!")
        if torch.cuda.is_available():
            self.aligner_model = self.aligner_model.cuda()
        self.bert_util = BertUtil(shortcut='roberta-large')

    def calculate_alignment(self, ref, can, spans=None):
        
        temp, temp_len, temp_ids, temp_att, temp_schema_idx, primary_span, secondary_span = self.bert_util.tokenize_sentence(ref, spans=spans)
        pos, pos_len, pos_ids, pos_att, pos_idx = self.bert_util.tokenize_sentence(can)

        temp_ids = torch.LongTensor(temp_ids).unsqueeze(0).to('cuda')
        pos_ids = torch.LongTensor(pos_ids).unsqueeze(0).to('cuda')

        temp_len_tensor = torch.tensor(temp_len).unsqueeze(0).to('cuda')
        pos_len_tensor = torch.tensor(pos_len).unsqueeze(0).to('cuda')

        alignment_matrix, cls_sim = self.aligner_model(temp_ids, temp_len_tensor, \
            positive_tensor=pos_ids, positive_lengths=pos_len_tensor, mode='eval')
        

        return alignment_matrix,cls_sim, temp, pos, primary_span, secondary_span

    def split_tokens(self, tokens, lengths):
        assert len(tokens) == sum(lengths) + 3
        tokens1 = tokens[1:1 + lengths[0]]
        tokens2 = tokens[2 + lengths[0]:-1]
        return tokens1, tokens2

    def link_example_schema(self, example):
        ret = self.schema_linker.link_example(example)
        return ret


if __name__ == '__main__':
    bert_aligner = BertAligner()
    # examples = json.load(open('data/spider/train_spider.json', 'r', encoding='utf-8'))
    # nl, restatement = bert_aligner.load_example(examples[0])
    # predictions = []
    # targets = []
    # with open('../results/qscwe/results/dev.sim', 'r') as f:
    #     predictions = f.readlines()
    # with open('../results/dev.target', 'r') as f:
    #     targets = f.readlines()
    # print(len(predictions))
    # print(len(targets))
    # pre, tar = (predictions[0], targets[0])
    # print(pre)
    # print(tar)
    # alignment_matrix, ids, tokens, lengths = bert_aligner.calculate_alignment(pre, tar)
    # print(alignment_matrix)
    # print(ids)
    # print(tokens)
    # print(lengths)
    # tokens1, tokens2 = bert_aligner.split_tokens(tokens, lengths)
    # print(tokens1)
    # print(tokens2)
    # alignment_matrix = alignment_matrix.squeeze(0).detach().cpu().numpy()
    # print(len(alignment_matrix))
    # print(len(alignment_matrix[0]))