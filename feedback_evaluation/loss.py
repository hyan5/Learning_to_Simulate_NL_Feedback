# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class HingeLoss(nn.Module):
    def __init__(self, margin=0.6, aggregation='max', l1_norm_weight=0.1, pos_prior_weight=0, neg_prior_weight=0, similarity_mode='average'):
        super(HingeLoss, self).__init__()
        self.margin = margin
        self.aggregation = aggregation
        self.similarity_mode = similarity_mode
        self.l1_norm_weight = l1_norm_weight
        self.pos_prior_weight = pos_prior_weight
        self.neg_prior_weight = neg_prior_weight

    def forward(self, pos_align, neg_align, lengths, pos_prior, neg_prior):
        template_lengths, positive_lengths, negative_lengths = lengths
        prior_temp_pos_mat, prior_temp_pos_mask_mat = pos_prior
        prior_temp_neg_mat, prior_temp_neg_mask_mat = neg_prior

        src_lengths = template_lengths
        pos_tgt_lengths = positive_lengths
        neg_tgt_lengths = negative_lengths

        positive_n = sum(template_lengths * positive_lengths)
        negative_n = sum(template_lengths * negative_lengths)

        pos_l1_norm, neg_l1_norm = torch.norm(pos_align, p=1) / positive_n, torch.norm(neg_align, p=1) / negative_n
        # pos_l1_norm, neg_l1_norm = torch.sum(torch.norm(pos_align, p='nuc', dim=(1,2))) / pos_align.size(0), torch.sum(torch.norm(neg_align, p='nuc', dim=(1,2))) / neg_align.size(0)
        
        """
        Prior Alignment
        """
        pos_prior_mat, neg_prior_mat = pos_align - prior_temp_pos_mat, neg_align - prior_temp_neg_mat
        pos_prior_mat, neg_prior_mat = pos_prior_mat * pos_prior_mat * prior_temp_pos_mask_mat, neg_prior_mat * neg_prior_mat * prior_temp_neg_mask_mat
        pos_prior, neg_prior = torch.sum(pos_prior_mat) / pos_align.size(0), torch.sum(neg_prior_mat) / neg_align.size(0)

        if self.aggregation == 'max':
            if self.similarity_mode == 'recall':
                pos_align_score, neg_align_score = torch.max(pos_align, -1)[0], torch.max(neg_align, -1)[0]
            elif self.similarity_mode == 'precision':
                pos_align_score, neg_align_score = torch.max(pos_align, 1)[0], torch.max(neg_align, 1)[0]
            else:
                pos_align_score, neg_align_score = (torch.max(pos_align, -1)[0], torch.max(pos_align, 1)[0]), (torch.max(neg_align, -1)[0], torch.max(neg_align, 1)[0])

        if self.similarity_mode != 'average' and self.similarity_mode != 'f1':
            pos_align_score = torch.sum(pos_align_score, -1)
            neg_align_score = torch.sum(neg_align_score, -1)

            if self.similarity_mode == 'recall':
                pos_align_score = torch.div(pos_align_score, src_lengths.float())
                neg_align_score = torch.div(neg_align_score, src_lengths.float())
            if self.similarity_mode == 'precision':
                pos_align_score = torch.div(pos_align_score, pos_tgt_lengths.float())
                neg_align_score = torch.div(neg_align_score, neg_tgt_lengths.float())
        else:
            pos_recall_score, pos_precision_score = pos_align_score
            neg_recall_score, neg_precision_score = neg_align_score

            pos_recall_score, pos_precision_score = torch.sum(pos_recall_score, -1), torch.sum(pos_precision_score, -1)
            neg_recall_score, neg_precision_score = torch.sum(neg_recall_score, -1), torch.sum(neg_precision_score, -1)

            pos_recall_score, pos_precision_score = torch.div(pos_recall_score, src_lengths.float()), torch.div(pos_precision_score, pos_tgt_lengths.float())
            neg_recall_score, neg_precision_score = torch.div(neg_recall_score, src_lengths.float()), torch.div(neg_precision_score, neg_tgt_lengths.float())
            if self.similarity_mode == 'f1':
                pos_align_score = torch.div(torch.mul(torch.mul(pos_recall_score, pos_precision_score), 2), pos_recall_score + pos_precision_score)
                neg_align_score = torch.div(torch.mul(torch.mul(neg_recall_score, neg_precision_score), 2), neg_recall_score + neg_precision_score)
            else:
                pos_align_score = torch.div(pos_recall_score + pos_precision_score, 2)
                neg_align_score = torch.div(neg_recall_score + neg_precision_score, 2)

        pure_ranking_loss = torch.mean(torch.clamp(self.margin - (pos_align_score - neg_align_score), min=0.0)) 
        l1_term = self.l1_norm_weight * (pos_l1_norm + neg_l1_norm)
        pos_prior *= self.pos_prior_weight
        neg_prior *= self.neg_prior_weight
        

        hinge_loss = pure_ranking_loss + l1_term + pos_prior + neg_prior
        
        return hinge_loss, pure_ranking_loss, l1_term, pos_prior, neg_prior