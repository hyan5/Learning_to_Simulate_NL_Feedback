# coding: utf-8
from curses.ascii import isdigit
import os
import re
import json
import dill
import torch
import random
import logging

import numpy as np
import fuzzymatch as fm

from tqdm import tqdm
from nltk.tokenize import word_tokenize
from unidecode import unidecode
from transformers import AutoTokenizer #BertTokenizerFast, RobertaTokenizerFast
from torch.utils.data import Dataset, DataLoader
import pdb


logger = logging.getLogger(__name__)
random.seed(1229)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class BertUtil:
    def __init__(self, shortcut='bert-base-uncased'):
        self.tokenizer_shortcut = shortcut
        # if 'bert-base' in shortcut:
        #     self.tokenizer = BertTokenizerFast.from_pretrained(shortcut)
        # elif 'roberta' in shortcut:
        #     self.tokenizer = RobertaTokenizerFast.from_pretrained(shortcut, add_prefix_space=True)
        self.tokenizer = AutoTokenizer.from_pretrained(shortcut, use_fast=True, add_prefix_space=True )

    def tokenize_sentence(self, sentence, schema_idx=None, spans=None):
        if not isinstance(sentence, list):
            sentence = sentence.split()

        tokenized_sentence = self.tokenizer(sentence, is_split_into_words=True, return_offsets_mapping=True)
        ids = tokenized_sentence['input_ids']

        tokens = [self.tokenizer.decode(token) for token in ids]

        offsets = tokenized_sentence['offset_mapping']
        updated_schema_idx = []
        updated_primary_span = []
        updated_secondary_span = []


        count = -1
        tokenized_span = [0] * len(sentence)

        for offset in offsets:
            if offset[0] == 0:
                if offset[1] == 0:
                    continue
                else:
                    count += 1
                    tokenized_span[count] = 1 if count == 0 else tokenized_span[count-1] + 1
            else:
                tokenized_span[count] += 1

        if schema_idx:
            for j, idx in enumerate(schema_idx):
                curr_idx = []
                for start, end in idx:
                    new_start = tokenized_span[start - 1] if start != 0 else 0
                    new_end = tokenized_span[end] - 1
                    curr_idx.append((new_start, new_end))
                updated_schema_idx.append(curr_idx)
        
        if spans:
            primary, secondary = spans
            for j, idx in enumerate(primary):
                start, end = idx
                new_start = tokenized_span[start - 1] if start != 0 else 0
                new_end = tokenized_span[end] - 1
                updated_primary_span.append((new_start, new_end))
                
            for j, idx in enumerate(secondary):
                start, end = idx
                new_start = tokenized_span[start - 1] if start != 0 else 0
                new_end = tokenized_span[end] - 1
                updated_secondary_span.append((new_start, new_end))
        if spans:
            return tokens, len(ids)-2, tokenized_sentence['input_ids'], np.array(tokenized_sentence['attention_mask']), updated_schema_idx, updated_primary_span, updated_secondary_span
        return tokens, len(ids)-2, tokenized_sentence['input_ids'], np.array(tokenized_sentence['attention_mask']), updated_schema_idx

    def tokens_to_ids(self, sentence_tokens):
        return self.tokenizer.convert_tokens_to_ids(sentence_tokens)

class AlignDataset(Dataset):
    def __init__(self, table_file, data_file, n_random, n_changing, n_swapping, n_dropping, data_type='train', negative_sampling_mode='random', tokenizer_shortcut='bert-base-uncased'):
        self.n_random = n_random
        self.n_changing = n_changing
        self.n_swapping = n_swapping
        self.n_dropping = n_dropping
        self.data_type = data_type
        self.db_id_key = 'db_id'
        self.bert_util = BertUtil(tokenizer_shortcut)
        self.negative_mode = negative_sampling_mode
        # self.agg_mapping = {'min': ['minimum'], 'minimun': ['min'], 'max': ['maximum', 'highest'], 'maximum': ['max', 'highest'], 'highest': ['max', 'maximum'], \
        #                'average': ['avg', 'mean'], 'avg': ['average', 'mean'], 'mean': ['avg', 'average'], 'sumation': ['sum'], 'sum': ['sumation'], \
        #                'greater than':['more than'], 'more than': ['greater than'], 'less than': ['smaller than'], 'smaller than': ['less than'], \
        #                'equals': ['equal'], 'equal': ['equals']}
        self.db_infos, self.examples = self.load_data_file(table_file, data_file)
        self.training_pairs, self.neg_len, self.neg_tags = self.build_training_data()

        logger.info(f'{self.__len__()} examples build')

    @staticmethod
    def load_data_file(db_file, data_file):
        raise NotImplementedError

    def build_training_data(self):
        if os.path.exists('cache'):
            if os.path.exists(f'cache/{self.data_type}_{self.negative_mode}_{self.n_random}_{self.n_changing}_{self.n_swapping}_{self.n_dropping}.bin'):
                train_triples = dill.load(open(f'cache/{self.data_type}_{self.negative_mode}_{self.n_random}_{self.n_changing}_{self.n_swapping}_{self.n_dropping}.bin', 'rb'))
                print(f'{self.data_type} data exists: {len(train_triples)}')
                # print(train_triples[0])
                # if data_type == 'train':
                #     return train_triples, None
                # else:
                neg_len = dill.load(open(f'cache/{self.data_type}_{self.negative_mode}_{self.n_random}_{self.n_changing}_{self.n_swapping}_{self.n_dropping}_neg_len.bin', 'rb'))
                print("Len Neg: ", len(neg_len))
                # print(masked)
                neg_tag = dill.load(open(f'cache/{self.data_type}_{self.negative_mode}_{self.n_random}_{self.n_changing}_{self.n_swapping}_{self.n_dropping}_neg_tag.bin', 'rb'))
                print("Len Tag: ", len(neg_tag))

                return train_triples, neg_len, neg_tag
        else:
            os.makedirs('cache')
            
        train_triples, neg_len, neg_tag = self.build_negative_from_positive(self.db_infos, self.examples, self.n_random, self.n_changing, self.n_swapping, self.n_dropping, mode=self.negative_mode)
        print(f'Triples: {len(train_triples)}')
        print(f'Neg Len: {len(neg_len)}')
        print(f'Neg Tag Len: {len(neg_tag)}')
        # print('First negative example: ', train_triples[0])
        # print('First 10 neg len value: ', masked_all[:])
        return train_triples, neg_len, neg_tag

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        
        text = self.training_pairs[idx]
        # read triplets (temp, pos, neg)
        template, positive, negative = text

        template_feedback, template_schema, template_schema_idx = template
        positive_feedback, positive_schema, positive_schema_idx = positive
        negative_feedback, negative_schema, negative_schema_idx = negative
     
        # tokenize all feedbcaks
        template_tokens, template_length, template_ids, template_attention_mask, template_schema_idx = self.bert_util.tokenize_sentence(template_feedback, template_schema_idx)
        positive_tokens, positive_length, positive_ids, positive_attention_mask, positive_schema_idx = self.bert_util.tokenize_sentence(positive_feedback, positive_schema_idx)
        negative_tokens, negative_length, negative_ids, negative_attention_mask, negative_schema_idx = self.bert_util.tokenize_sentence(negative_feedback, negative_schema_idx)
        # print(f'temp tokens: {template_tokens}, {template_length}\npos tokens: {positive_tokens}, {positive_length}\nneg tokens: {negative_tokens}, {negative_length}')
        # convert tokens to ids
        template_ids = torch.LongTensor(template_ids)
        positive_ids = torch.LongTensor(positive_ids)
        negative_ids = torch.LongTensor(negative_ids)
        # print(f'temp ids: {template_ids}\npos ids: {positive_ids}\nneg ids: {negative_ids}')

        # get stopwords weight matrix
        template_weight_mat = self.get_weight_mask_matrix(template_tokens, template_length)
        positive_weight_mat = self.get_weight_mask_matrix(positive_tokens, positive_length)
        negative_weight_mat = self.get_weight_mask_matrix(negative_tokens, negative_length)
        # print(f'temp weight: {template_weight_mat}, {template_weight_mat.shape}\npos weight: {positive_weight_mat}, {positive_weight_mat.shape}\nneg weight: {negative_weight_mat}, {negative_weight_mat.shape}')

        return (template_ids, positive_ids, negative_ids), (template_weight_mat, positive_weight_mat, negative_weight_mat), \
               (template_tokens, positive_tokens, negative_tokens), (template_length, positive_length, negative_length), (template_attention_mask, positive_attention_mask, negative_attention_mask), \
               (template_schema, positive_schema, negative_schema), (template_schema_idx, positive_schema_idx, negative_schema_idx)

    @staticmethod
    def get_weight_mask_matrix(tokens, length): # col_stopwords=STOP_WORD_LIST, row_stopwords=TEMPLATE_KEYWORDS
        col_tokens = tokens[1: 1 + length]
        weight_matrix = np.ones(length)
        # for idx, col_token in enumerate(col_tokens):
        #     if col_token in col_stopwords:
        #         weight_matrix[idx, :] = 0.5
        # for idx, row_token in enumerate(row_tokens):
        #     if row_token in row_stopwords:
        #         weight_matrix[:, idx] = 0.5
        return weight_matrix

    def find_sublist(self, sent, schema_info):
        results = []
        sent = sent.split()
        if schema_info.replace(' ', '') in sent:
            schema_info = schema_info.replace(' ', '')
        schema_info = schema_info.split()
        len_schema = len(schema_info)
        for ind in (i for i,e in enumerate(sent) if e == schema_info[0]):
            if sent[ind : ind+len_schema] == schema_info:
                results.append((ind, ind+len_schema-1))

        return results

    def find_position(self, can_sent, schema_list):
        sort_list = schema_list.copy()
        sort_list.sort(key=lambda x: len(x.split()), reverse=True)
        matched_idx = {}
        results = []
        for sc in sort_list:
            sc_idx = self.find_sublist(can_sent, sc)
            # if len(sc.split()) != 1:
            #     matched_idx[sc] = sc_idx
            # else:
            idx_sc = []
            for start, end in sc_idx:
                is_overlap = False
                for curr_idx in matched_idx:
                    if is_overlap:
                        break
                    for curr_start, curr_end in matched_idx[curr_idx]:
                        if start >= curr_start and start <= curr_end:
                            is_overlap = True
                            break
                if not is_overlap:
                    idx_sc.append((start, end))

            matched_idx[sc] = idx_sc
        # pdb.set_trace()
        for sc in schema_list:
            results.append(matched_idx[sc])
        return results

    def idx_ch_to_token(self, sent, ch_idx):
        token_mapping = {}
        curr_idx = 0
        results = []
        for i, ch in enumerate(sent):
            if ch == ' ':
                curr_idx += 1
                continue
            token_mapping[str(i)] = curr_idx

        for idx in ch_idx:
            curr = []
            for start, end in idx:
                curr.append((token_mapping[str(start)], token_mapping[str(end)]))
            results.append(curr)
        return results

    def eliminate_overlap(self, schema_idx_dict):
        sorted_list = sorted(schema_idx_dict.items(), key=lambda x: len(x[0].split()), reverse=True)
        res_schema = []
        res_idx = []
        for i, item in enumerate(sorted_list):
            sc, idx = item
            can_idx = []
            for start, end in idx:
                is_overlap = False
                for curr_idx in res_idx:
                    if is_overlap:
                        break
                    for curr_start, curr_end in curr_idx:
                        if start >= curr_start and start <= curr_end:
                            is_overlap = True
                if not is_overlap:
                    can_idx.append((start, end))

            if can_idx:
                res_schema.append(sc)
                res_idx.append(can_idx)

        return res_schema, res_idx

    def find_position_num(self, can_list, ref_feedback, can_feedback=None):
        matched_num = []
        ref_matched_num_idx = []
        ref = ref_feedback.split()
        can_ref_idx = self.find_position(ref_feedback, can_list)

        if can_feedback:
            can_matched_num_idx = []
            can = can_feedback.split()
            can_can_idx = self.find_position(can_feedback, can_list)

        for i, num in enumerate(can_list):
            idx_ref = []
            for start, end in can_ref_idx[i]:
                if 'step' in ref[start-1] or 'and' in ref[start-1]:
                    pass
                else:
                    idx_ref.append((start, end))

            if can_feedback:
                idx_can = []
                for start, end in can_can_idx[i]:
                    if 'step' in can[start-1]  or 'and' in can[start-1]:
                        pass
                    else:
                        idx_can.append((start, end))
            
                if idx_ref and idx_can:
                    matched_num.append(num)
                    ref_matched_num_idx.append(idx_ref)
                    can_matched_num_idx.append(idx_can)
            else:
                ref_matched_num_idx.append(idx_ref)
        
        if can_feedback:
            return matched_num, ref_matched_num_idx, can_matched_num_idx
        else:
            return ref_matched_num_idx

    def replace_original_modified_schema(self, sent, sc_ori, sc):
        ori_position = self.find_sublist(sent, sc_ori)
        sent = sent.split()
        _sent = sent.copy()
        for start, end in ori_position:
            sent[start:end+1] = sc.split()
        # pdb.set_trace()
        return ' '.join(sent)
    # def find_position_agg(self, agg_matched, temp_feedback, pos_feedback):


    def build_negative_from_positive(self, db_infos, examples, n_random, n_changing, n_swapping, n_dropping, mode='random'):
        assert mode in ('random', 'replace', 'mix')

        if mode == 'mix':
            mode = 'random replace'

        ret = []
        neg_len = []
        neg_tags = []

        out_json = []
        
        replace_zeros = []
        
        character_set = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        logger.info('Generating negative examples...')
        # count = 0
        # smaller_set = []
        step = 1 #527 if self.data_type == 'train' else 174
        # largaer_than_one = 0
        count = 0
        fuzzy_count = 0
        # Randomly Sampled Data
        sampled = [random.randint(0, 7639) for i in range(50)]
        sampled_matching = []
        print(sampled)

        for i in tqdm(range(0, len(examples), step)):
            # if i not in sampled:
            #     continue
            sampled_data = {}
            feedback = examples[i]['template_feedback'].strip().lower()
            pos_example = ' '.join(word_tokenize(examples[i]['feedback'].strip().lower())).replace("``", "\"").replace("''", "\"")
            edits = [edit.strip() for edit in examples[i]['edits']]

            # primary_span = examples[i]['primary_span']
            # secondary_span = examples[i]['secondary_span']

            # question = examples[i]['question']
            # gold_parse = examples[i]['gold_parse']
            # wrong_parse = unidecode(examples[i]['predicted_parse_with_values'])
            # wrong_explanation = [unidecode(exp) for exp in examples[i]['predicted_parse_explanation']]
            # original_edits = examples[i]['edits_original']

            neg_tag = [0, 0, 0, 0]

            _feedback = feedback
            _pos_example = pos_example
            _edits = edits

            neg_examples = []
            out_exp = {'db_id': examples[i]['db_id'], 'tabel_names': db_infos[examples[i]['db_id']]['table_names'], 'column_names': [table[1] for table in db_infos[examples[i]['db_id']]['column_names'][1:]], 
                        'template_feedback': examples[i]['template_feedback'], 'primary_span': examples[i]['primary_span'], 'secondary_span':examples[i]['secondary_span'], 'positive_example': examples[i]['feedback'], 'negative_examples': [], 'neg_len': 0}

            if 'replace' in mode:
                db_id = examples[i]['db_id']
                table_names_original = db_infos[db_id]['table_names_original']
                column_names_original = [col_name[1] for col_name in db_infos[db_id]['column_names_original'][1:]]
                schema_list_original = table_names_original + column_names_original

                table_names = db_infos[db_id]['table_names']
                column_names = [col_name[1] for col_name in db_infos[db_id]['column_names'][1:]]
                schema_list = table_names + column_names

                assert len(schema_list_original) == len(schema_list)
                schema_dict = dict(zip(schema_list_original, schema_list))

                # agg_opt = ['maximum', 'minimum', 'average', 'sumation', 'max', 'min', 'avg', 'sum', 'number of', 'mean', 'highest']
                # agg_comp = ['greater than', 'more than', 'less than', 'smaller than', 'equals', 'equal']

                """
                Schema Matching
                """

                # Extract schema info from positive feedback via fuzzy string macting
                fuzzy_schema_scores = fm.get_matched_entries(pos_example, list(set(table_names + column_names)), m_theta=0.85, s_theta=0.85)
                
                fuzzy_matched = []
                fuzzy_matched_ch_idx = []
                fuzzy_matched_idx = []
                matched_score = []


                for sc, match in fuzzy_schema_scores:
                    fuzzy_matched.append(sc)
                    fuzzy_matched_ch_idx.append([idx[2] for idx in match])
                    matched_score += [idx[1] for idx in match]


                fuzzy_matched_idx = self.idx_ch_to_token(pos_example, fuzzy_matched_ch_idx)
                # _fuzzy_matched_idx = self.find_position(pos_example, fuzzy_matched)
                # fuzzy_matched_size = [len(idx) for idx in fuzzy_matched_idx]
                # _fuzzy_matched_size = 0
                # for id, sc in enumerate(fuzzy_matched):
                #     if sc in schema_list:
                #         _fuzzy_matched_size += len(_fuzzy_matched_idx[id])
                #     else:
                #         _fuzzy_matched_size += len(fuzzy_matched_idx[id])

                # if fuzzy_matched_size != _fuzzy_matched_size:
                #     fuzzy_count += 1
                

                fuzzy_matched, fuzzy_matched_idx = self.eliminate_overlap(dict(zip(fuzzy_matched, fuzzy_matched_idx)))
                
                # Extract schema info from edits
                # edits_schema_original = []
                edits = ' '.join(edits)
                edits_schema = []

                # for sc in schema_dict:
                #     if sc.lower() in edits.split():
                #         edits_schema.append(schema_dict[sc])
                #         edits_schema_original.append(sc)

                for sc in schema_list:
                    if sc.lower() in edits:
                        if sc not in edits_schema:
                            edits_schema.append(sc)
                        # edits_schema_original.append(sc)
                
                
                edits_schema_idx = self.find_position(feedback, edits_schema)
                edits_schema, edits_schema_idx = self.eliminate_overlap(dict(zip(edits_schema, edits_schema_idx)))
                # Calculate the intersection of edits schema and fuzzy matched schema
                fuzzy_schema_idx_dict = dict(zip(fuzzy_matched, fuzzy_matched_idx))
                edits_schema_idx_dict = dict(zip(edits_schema, edits_schema_idx))

                matched_schema = [schema for schema in edits_schema if schema in fuzzy_matched]                
                temp_schema_idx = [edits_schema_idx_dict[schema] for schema in matched_schema]
                pos_schema_idx = [fuzzy_schema_idx_dict[schema] for schema in matched_schema]

                if not matched_schema:
                    count += 1

                """
                Multi-Sentence Indicator
                """
                # Current feedback is a multi-sentece feedback if sent_count >= 2
                sent_count = 0
                for sent in pos_example.split('.'):
                    # print(sent)
                    sent = sent.strip()
                    # pdb.set_trace()
                    if sent and sent[0] in character_set and len(sent.split()) >= 2:
                        sent_count += 1
                    else:
                        continue
                printFlag = False
                # if printFlag:
                #     count += 1
                #     sampled_data['index'] = i
                #     sampled_data['question'] = question
                #     sampled_data['gold_parse'] = gold_parse
                #     sampled_data['predict_parse'] = wrong_parse
                #     sampled_data['explanation'] = wrong_explanation
                #     sampled_data['original_edits'] = original_edits
                #     sampled_data['pure_edits'] = _edits
                #     sampled_data['original_template_feedback'] = _feedback
                #     sampled_data['matched_edits_schema_original'] = edits_schema_orginal
                #     sampled_data['template_feedback_remove_underscore_lower_case'] = feedback
                #     sampled_data['matched_edits_schema'] = edits_schema
                #     sampled_data['matched_edits_schema_idx'] = edits_schema_idx
                #     sampled_data['positive_feedback'] = pos_example
                #     sampled_data['fuzzy_matching_scores'] = list(fuzzy_schema_scores)
                #     sampled_data['fuzzy_matched_schema'] = fuzzy_matched
                #     sampled_data['fuzzy_matched_schema_idx'] = fuzzy_matched_idx
                #     sampled_data['intersection_matched_schema'] = matched_schema
                #     sampled_data['temp_matched_schema_idx'] = temp_schema_idx
                #     sampled_data['pos_matched_schema_idx'] = pos_schema_idx

                #     print(f'Index: {i}')
                #     print(f'Question: {question}')
                #     print(f'Original Edits: {original_edits}')
                #     print(f'Pure Edits: {_edits}')
                #     print(f'Original Template Feedback: {_feedback}')
                #     print(f'Edits Schema Original: {edits_schema_orginal}')
                #     print(f'Edits Schema: {edits_schema}')
                #     print(f'Edits Schema Idx: {edits_schema_idx}')
                #     print(f'Template Feedback: {feedback}')
                #     print(f'Positive Feedback: {pos_example}')
                #     print(f'Fuzzy Matching Scores: {fuzzy_schema_scores}')
                #     print(f'Fuzzy Schema: {fuzzy_matched}')
                #     print(f'Fuzzy Schema Idx: {fuzzy_matched_idx}')
                #     print(f'Matched Schema: {matched_schema}')
                #     print(f'Template Matched Schema Idx: {temp_schema_idx}')
                #     print(f'Positive Matched Schema Idx: {pos_schema_idx}')

                #     print(f'Table Names: {table_names}')
                #     print(f'Column Names: {column_names}')
                    # print(f'Gold Parse: {gold_parse}')
                    # print(f'Wrong Parse: {wrong_parse}')
                    # print(f'Wrong Explanation: {wrong_explanation}')
                    # print()
                    # sampled_matching.append(sampled_data)
               
                
                neg_exp = []
                neg_exp_schema = []

                tab_rep = []
                col_rep = []
                tab_col_rep = []

                tab_rep_schema = []
                col_rep_schema = []
                tab_col_rep_schema = []

                tab_rep_schema_idx = []
                col_rep_schema_idx = []
                tab_col_rep_schema_idx = []

                swapping = []
                swapping_schema = []
                swapping_schema_idx = []

                dropping = []
                dropping_schema = []
                dropping_schema_idx = []

                """
                Replace Schema Info
                """
                for idx, schema in enumerate(matched_schema):
                    # Replace table name
                    if schema in table_names:
                        for t_name in table_names:
                            if t_name not in matched_schema:
                                neg = pos_example.split()
                                for start, end in pos_schema_idx[idx]:
                                    # rep_list = []
                                    # rep_list.append(t_name)
                                    
                                    neg[start : end+1] = [t_name]

                                neg = ' '.join(neg).strip()

                                if neg != pos_example and neg not in tab_rep:
                                    neg_matched_schema = matched_schema.copy()
                                    neg_matched_schema[idx] = t_name
                                    tab_rep.append(neg)
                                    tab_rep_schema.append(neg_matched_schema)
                                    tab_rep_schema_idx.append(self.find_position(neg, neg_matched_schema))

                    # Replace column name
                    if schema in column_names:
                        for c_name in column_names:
                            if c_name not in matched_schema:
                                neg = pos_example.split()
                                for start, end in pos_schema_idx[idx]:
                                    neg[start : end+1] = [c_name]
                                
                                neg = ' '.join(neg).strip()

                                if neg != pos_example and neg not in col_rep:
                                    neg_matched_schema = matched_schema.copy()
                                    neg_matched_schema[idx] = c_name
                                    col_rep.append(neg)
                                    col_rep_schema.append(neg_matched_schema)
                                    col_rep_schema_idx.append(self.find_position(neg, neg_matched_schema))

                # Replace both table and column schema
                for idx_exp, exp in enumerate(tab_rep):
                    for idx, schema in enumerate(tab_rep_schema[idx_exp]):
                        if schema in column_names:
                            for c_name in column_names:
                                if c_name not in matched_schema:
                                    neg = exp.split()
                                    for start, end in tab_rep_schema_idx[idx_exp][idx]:
                                        # rep_list = []
                                        # rep_list.append(c_name)
                                        # for num_ele in range(end - start):
                                        #     rep_list.append('[placeholder]')
                                        neg[start : end+1] = [c_name]
                                    # neg = [token for token in neg if token != '[placeholder]']
                                    neg = ' '.join(neg).strip()

                                    if neg != pos_example and neg not in tab_col_rep:
                                        neg_matched_schema = tab_rep_schema[idx_exp].copy()
                                        neg_matched_schema[idx] = c_name
                                        tab_col_rep.append(neg)
                                        tab_col_rep_schema.append(neg_matched_schema)
                                        tab_col_rep_schema_idx.append(self.find_position(neg, neg_matched_schema))
                """
                Swapping
                """
                if len(matched_schema) > 1:
                    for idx, schema in enumerate(matched_schema):
                        for idx_rep in range(idx + 1, len(matched_schema)):
                            neg = pos_example.split()
                            for start, end in pos_schema_idx[idx_rep]:
                                neg[start:end+1] = [schema]
                                # if end - start == 0:
                                #     neg[start] = schema
                                # else:
                                #     rep_list = []
                                #     rep_list.append(schema)
                                #     for num_ele in range(end - start):
                                #         rep_list.append('[placeholder]')
                                #     neg[start : end+1] = rep_list

                            for start, end in pos_schema_idx[idx]:
                                # if end - start == 0:
                                #     neg[start] = matched_schema[idx_rep]
                                # else:
                                #     rep_list = []
                                #     rep_list.append(matched_schema[idx_rep])
                                #     for num_ele in range(end - start):
                                #         rep_list.append('[placeholder]')
                                neg[start : end+1] = [matched_schema[idx_rep]]
                            # neg = [token for token in neg if token != '[placeholder]']
                            neg = ' '.join(neg).strip()

                            if neg != pos_example and neg not in swapping:
                                neg_matched_schema = matched_schema.copy()
                                neg_matched_schema[idx] = matched_schema[idx_rep]
                                neg_matched_schema[idx_rep] = schema
                                swapping.append(neg)
                                swapping_schema.append(neg_matched_schema)
                                swapping_schema_idx.append(self.find_position(neg, neg_matched_schema))
                
                """
                Dropping
                """
                if sent_count >= 2:
                    sents = [sent.strip() for sent in pos_example.split('.') if sent and sent.strip()[0] in character_set]
                    neg = sents[0]
                    if neg != pos_example and neg not in dropping:
                        dropping.append(neg)
                        dropping_schema_idx.append(self.find_position(neg, matched_schema))
                    neg = sents[-1]
                    if neg != pos_example and neg not in dropping:
                        dropping.append(neg)
                        dropping_schema_idx.append(self.find_position(neg, matched_schema))
                    if len(sents) == 3:
                        neg = sents[1]
                        if neg != pos_example and neg not in dropping:
                            dropping.append(neg)
                            dropping_schema_idx.append(self.find_position(neg, matched_schema))
                elif len(pos_example.split()) >= 14:
                    sents = pos_example.split()
                    len_split = int(len(sents) / 2)
                    neg = ' '.join(sents[:len_split])
                    if neg != pos_example and neg not in dropping:
                        dropping.append(neg)
                        dropping_schema_idx.append(self.find_position(neg, matched_schema))
                    neg = ' '.join(sents[len_split:])
                    if neg != pos_example and neg not in dropping:
                        dropping.append(neg)
                        dropping_schema_idx.append(self.find_position(neg, matched_schema))

                phrase_changing_neg = []
                phrase_changing_schema = []
                phrase_changing_schema_idx = []

                phrase_swapping_neg = []
                phrase_swapping_schema = []
                phrase_swapping_schema_idx = []

                phrase_dropping_neg = []
                phrase_dropping_schema = []
                phrase_dropping_schema_idx = []

                phrase_changing_all = tab_rep + col_rep + tab_col_rep
                phrase_changing_all_schema = tab_rep_schema + col_rep_schema + tab_col_rep_schema
                phrase_changing_all_schema_idx = tab_rep_schema_idx + col_rep_schema_idx + tab_col_rep_schema_idx

                for j, neg in enumerate(phrase_changing_all):
                    if neg not in phrase_changing_neg:
                        phrase_changing_neg.append(neg)
                        phrase_changing_schema.append(phrase_changing_all_schema[j])
                        phrase_changing_schema_idx.append(phrase_changing_all_schema_idx[j])


                for j, neg in enumerate(swapping):
                    if neg not in phrase_swapping_neg:
                        phrase_swapping_neg.append(neg)
                        phrase_swapping_schema.append(swapping_schema[j])
                        phrase_swapping_schema_idx.append(swapping_schema_idx[j])
                
                for j, neg in enumerate(dropping):
                    if neg not in phrase_dropping_neg:
                        phrase_dropping_neg.append(neg)
                        phrase_dropping_schema.append(matched_schema)
                        phrase_dropping_schema_idx.append(dropping_schema_idx[j])

                phrase_changing = list(zip(phrase_changing_neg, phrase_changing_schema,  phrase_changing_schema_idx))
                phrase_swapping = list(zip(phrase_swapping_neg, phrase_swapping_schema, phrase_swapping_schema_idx))
                phrase_dropping = list(zip(phrase_dropping_neg, phrase_dropping_schema, phrase_dropping_schema_idx))

                
                if len(phrase_changing) > self.n_changing:
                    phrase_changing = random.sample(phrase_changing, self.n_changing)
                if len(phrase_swapping) > self.n_swapping:
                    phrase_swapping = random.sample(phrase_swapping, self.n_swapping)
                if len(phrase_dropping) > self.n_dropping:
                    phrase_dropping = random.sample(phrase_dropping, self.n_dropping)

                neg_exp += phrase_changing
                neg_exp += phrase_swapping
                neg_exp += phrase_dropping

                neg_tag[0] = len(phrase_changing)
                neg_tag[1] = len(phrase_swapping)
                neg_tag[2] = len(phrase_dropping)

                if not neg_exp:
                    replace_zeros.append({'db_id': db_id, 'table_names': table_names, 'column_names': column_names, 'template_feedback': feedback, 'positive_example': pos_example})
                else:
                    with open(f'cache/{self.data_type}_phrase_changing.txt', 'a') as f:
                        f.write(str(len(phrase_changing)) + '\n')
                    with open(f'cache/{self.data_type}_phrase_swapping.txt', 'a') as f:
                        f.write(str(len(phrase_swapping)) + '\n')
                    with open(f'cache/{self.data_type}_phrase_dropping.txt', 'a') as f:
                        f.write(str(len(phrase_dropping)) + '\n')

            
                for neg, neg_schema, neg_schema_idx in neg_exp:
                    # pdb.set_trace()
                    out_exp['negative_examples'].append(neg)
                    neg_examples.append(((feedback, matched_schema, temp_schema_idx), \
                                         (pos_example, matched_schema, pos_schema_idx), \
                                         (neg, neg_schema, neg_schema_idx)))

            if 'random' in mode:
                neg_tag[3] = n_random
                neg_all = [example['feedback'] for example in examples[:i]] + [example['feedback'] for example in examples[i+1:]]
                neg_samples = random.sample(neg_all, n_random)
                # print('Random len: ', len(neg_samples))
                for sample in neg_samples:
                    neg_examples.append(((feedback, [], []), (pos_example, [], []), (sample, [], [])))
                out_exp['negative_examples'] += neg_samples


            if len(neg_examples) != 0:
                neg_len.append(len(neg_examples))
                neg_tags.append(neg_tag)
                out_exp['neg_len'] = len(neg_examples)
                out_json.append(out_exp)

            ret += neg_examples
            
            # print('Total neg: ', len(neg_examples))

        print(f'Total Negative Zeros Length: {len(replace_zeros)}')

        with open(f'cache/zeros_{self.data_type}_{self.negative_mode}_{self.n_random}_{self.n_changing}_{self.n_swapping}_{self.n_dropping}.json', 'w') as f:
            json.dump(replace_zeros, f, indent=4)
        with open(f'cache/negtive_examples_{self.data_type}_{self.negative_mode}_{self.n_random}_{self.n_changing}_{self.n_swapping}_{self.n_dropping}.json', 'w') as f:
            json.dump(out_json, f, indent=4)

        dill.dump(ret, open(f'cache/{self.data_type}_{self.negative_mode}_{self.n_random}_{self.n_changing}_{self.n_swapping}_{self.n_dropping}.bin', 'wb'))
        dill.dump(neg_len, open(f'cache/{self.data_type}_{self.negative_mode}_{self.n_random}_{self.n_changing}_{self.n_swapping}_{self.n_dropping}_neg_len.bin', 'wb'))
        dill.dump(neg_tags, open(f'cache/{self.data_type}_{self.negative_mode}_{self.n_random}_{self.n_changing}_{self.n_swapping}_{self.n_dropping}_neg_tag.bin', 'wb'))

        print(f'Count: {count}')

        # with open('schema_matching_sampled_data.json', 'w') as f:
        #     json.dump(sampled_matching, f, indent=4)
        
        return ret, neg_len, neg_tags


    def get_column_table_pairs(self, db_info):
        raise NotImplementedError

    @staticmethod
    def collate_fn(data):
        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths
        # print("data in collate_fn: " + str(data))
        tensors, weights, texts, lengths, att_weights, schemas, schema_idx = zip(*data)
        template_tensors, positive_tensors, negative_tensors = zip(*tensors)
        template_weights, positive_weights, negative_weights = zip(*weights)
        template_lengths, positive_lengths, negative_lengths = zip(*lengths)
        template_texts, positive_texts, negative_texts = zip(*texts)
        template_atts, positive_atts, negative_atts = zip(*att_weights)
        template_schema, positive_schema, negative_schema = zip(*schemas)
        template_schema_idx, positive_schema_idx, negative_schema_idx = zip(*schema_idx)
        batch_size = len(template_texts)

        # print(f'Template lengths: {template_lengths}')
        # print(f'Positive lengths: {positive_lengths}')
        # print(f'Negative lengths: {negative_lengths}')

        # print(f'Template Tensors: {template_tensors}')
        # print(f'Positive Tensors: {positive_tensors}')
        # print(f'Negative Tensors: {negative_tensors}')

        # tensors
        template_tensors = merge(template_tensors)[0]
        positive_tensors = merge(positive_tensors)[0]
        negative_tensors = merge(negative_tensors)[0]

        temp_max_len = max([_ for _ in template_lengths])
        pos_max_len = max([_ for _ in positive_lengths])
        neg_max_len = max([_ for _ in negative_lengths])


        # print(f'Template Tensors: {template_tensors}\nLengths: {[len(_) for _ in template_tensors]}')
        # print(f'Positive Tensors: {positive_tensors}\nLengths: {[len(_) for _ in positive_tensors]}')
        # print(f'Negative Tensors: {negative_tensors}\nLengths: {[len(_) for _ in negative_tensors]}')

        # weights
        template_matrix_shape = batch_size, temp_max_len
        positive_matrix_shape = batch_size, pos_max_len
        negative_matrix_shape = batch_size, neg_max_len

        # print(f'Template Matrix Shape: {template_matrix_shape}')
        # print(f'Positive Matrix Shape: {positive_matrix_shape}')
        # print(f'Negative Matrix Shape: {negative_matrix_shape}')
        
        template_weight_matrix = np.zeros(template_matrix_shape, dtype=np.float32)
        positive_weight_matrix = np.zeros(positive_matrix_shape, dtype=np.float32)
        negative_weight_matrix = np.zeros(negative_matrix_shape, dtype=np.float32)

        
        # print(f'Template Weight Matrix: {template_weight_matrix}')
        # print(f'Positive Weight Matrix: {positive_weight_matrix}')
        # print(f'Negative Weight Matrix: {negative_weight_matrix}')

        assert len(positive_weights) == len(negative_weights) == batch_size

        for idx, (template_weight, positive_weight, negative_weight) in enumerate(zip(template_weights, positive_weights, negative_weights)):
            template_weight_matrix[idx, :template_weight.shape[0]] = template_weight
            positive_weight_matrix[idx, :positive_weight.shape[0]] = positive_weight
            negative_weight_matrix[idx, :negative_weight.shape[0]] = negative_weight

        template_weight_matrix = torch.Tensor(template_weight_matrix)
        positive_weight_matrix = torch.Tensor(positive_weight_matrix)
        negative_weight_matrix = torch.Tensor(negative_weight_matrix)

        # print(f'Template Weight Matrix: {template_weight_matrix}')
        # print(f'Positive Weight Matrix: {positive_weight_matrix}')
        # print(f'Negative Weight Matrix: {negative_weight_matrix}')

        # attentation weights
        template_att_shape = batch_size, temp_max_len + 2
        positive_att_shape = batch_size, pos_max_len + 2
        negative_att_shape = batch_size, neg_max_len + 2

        # print(f'collate_fn positive_matrix_shape: {positive_matrix_shape}')
        # print(f'collate_fn negative_matrix_shape: {negative_matrix_shape}')
        template_att_matrix = np.zeros(template_att_shape, dtype=np.float32)
        positive_att_matrix = np.zeros(positive_att_shape, dtype=np.float32)
        negative_att_matrix = np.zeros(negative_att_shape, dtype=np.float32)
        
        for idx, (temp_att, pos_att, neg_att) in enumerate(zip(template_atts, positive_atts, negative_atts)):
            template_att_matrix[idx, :temp_att.shape[0]] = temp_att
            positive_att_matrix[idx, :pos_att.shape[0]] = pos_att
            negative_att_matrix[idx, :neg_att.shape[0]] = neg_att
        # print(f'positive_att_weight: {positive_att_weight}')
        # print(f'negative_att_weight: {negative_att_weight}')

        template_att_matrix = torch.Tensor(template_att_matrix)
        positive_att_matrix = torch.Tensor(positive_att_matrix)
        negative_att_matrix = torch.Tensor(negative_att_matrix)


        """
        Prior Alignment Matrix
        """
        prior_temp_pos_shape = batch_size, temp_max_len, pos_max_len
        prior_temp_neg_shape = batch_size, temp_max_len, neg_max_len

        prior_temp_pos_mat = np.zeros(prior_temp_pos_shape, dtype=np.float32)
        prior_temp_neg_mat = np.zeros(prior_temp_neg_shape, dtype=np.float32)

        prior_temp_pos_mask_mat = np.zeros(prior_temp_pos_shape, dtype=np.float32)
        prior_temp_neg_mask_mat = np.zeros(prior_temp_neg_shape, dtype=np.float32)

        # print(f'template_schema: {template_schema}')
        # print(f'template_schema_idx: {template_schema_idx}')

        # print(f'pos_schema: {positive_schema}')
        # print(f'pos_schema_idx: {positive_schema_idx}')

        # print(f'neg_schema: {negative_schema}')
        # print(f'neg_schema_idx: {negative_schema_idx}')

        for idx in range(batch_size):
            for i, sc in enumerate(template_schema[idx]):
                for start_row, end_row in template_schema_idx[idx][i]:
                    for start_col, end_col in positive_schema_idx[idx][i]:
                        prior_temp_pos_mat[idx, start_row : end_row+1, start_col : end_col+1] = 1
                        prior_temp_pos_mask_mat[idx, :, start_col : end_col+1] = 1

                    for start_col, end_col in negative_schema_idx[idx][i]:
                        if negative_schema[idx][i] == sc:
                            prior_temp_neg_mat[idx, start_row : end_row+1, start_col : end_col+1] = 1
                        else:
                            prior_temp_neg_mat[idx, start_row : end_row+1, start_col : end_col+1] = 0

                        prior_temp_neg_mask_mat[idx, :, start_col : end_col+1] = 1

                    prior_temp_pos_mask_mat[idx, start_row : end_row+1] = 1
                    prior_temp_neg_mask_mat[idx, start_row : end_row+1] = 1

        prior_temp_pos_mat = torch.Tensor(prior_temp_pos_mat)
        prior_temp_neg_mat = torch.Tensor(prior_temp_neg_mat)

        prior_temp_pos_mask_mat = torch.Tensor(prior_temp_pos_mask_mat)
        prior_temp_neg_mask_mat = torch.Tensor(prior_temp_neg_mask_mat)

        # print(f'Prior TP: {prior_temp_pos_mat}')
        # print(f'Prior TP Mask: {prior_temp_pos_mask_mat}')

        # print(f'Prior TN: {prior_temp_neg_mat}')
        # print(f'Prior TN Mask: {prior_temp_neg_mask_mat}')


        # lengths
        template_lengths = torch.LongTensor([_ for _ in template_lengths])
        positive_lengths = torch.LongTensor([_ for _ in positive_lengths])
        negative_lengths = torch.LongTensor([_ for _ in negative_lengths])

        return (template_tensors, positive_tensors, negative_tensors), (template_weight_matrix, positive_weight_matrix, negative_weight_matrix), \
               (template_lengths, positive_lengths, negative_lengths), (template_texts, positive_texts, negative_texts), (template_att_matrix, positive_att_matrix, negative_att_matrix), \
               (template_schema, positive_schema, negative_schema), (template_schema_idx, positive_schema_idx, negative_schema_idx), \
               (prior_temp_pos_mat, prior_temp_pos_mask_mat), (prior_temp_neg_mat, prior_temp_neg_mask_mat)


    def get_dataloader(self, batch_size, num_workers=4, shuffle=True):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=self.collate_fn), self.neg_len, self.neg_tags

class SpiderAlignDataset(AlignDataset):
    def __init__(self, table_file, data_file, n_random=0, n_changing=0, n_swapping=0, n_dropping=0, data_type='train', negative_sampling_mode='random', tokenizer_shortcut='bert-base-uncased'):
        super().__init__(table_file, data_file, n_random, n_changing, n_swapping, n_dropping, negative_sampling_mode=negative_sampling_mode, data_type=data_type, tokenizer_shortcut=tokenizer_shortcut)

    @staticmethod
    def load_data_file(db_file, data_file):
        db_infos = {_['db_id']: _ for _ in json.load(open(db_file, 'r', encoding='utf-8'))}
        examples = json.load(open(data_file, 'r', encoding='utf-8'))
        return db_infos, examples

    def get_column_table_pairs(self, db_info):
        table_names = db_info['table_names_original']
        column_table_pairs = [('_'.join(column_name.split()).lower(), table_names[table_idx].lower())
                              for table_idx, column_name in db_info['column_names_original'][1:]]
        return column_table_pairs

if __name__ == '__main__':

    # Data directory
    table_file = '../data/spider/tables.json'
    data_file = '../data/splash/train_w_edits.json'
    # data_file = '../data/splash/dev_w_edits.json'


    # for debug purpose
    data_type = 'train' if 'train' in data_file else 'dev'
    dataset = SpiderAlignDataset(table_file, data_file, n_random=0, n_changing=1, n_swapping=0, n_dropping=0, data_type=data_type, negative_sampling_mode='replace', tokenizer_shortcut='bert-base-uncased')

    # for i in range(7639):
    #     item = dataset.__getitem__(i)
    dataloader, neg_len, neg_tags = dataset.get_dataloader(batch_size=1, shuffle=False, num_workers=4)
    print(len(dataloader))
    for batch_data in dataloader:
        (template_tensors, positive_tensors, negative_tensors), (template_weight_matrix, positive_weight_matrix, negative_weight_matrix), \
        (template_lengths, positive_lengths, negative_lengths), (template_texts, positive_texts, negative_texts), (template_att_mat, positive_att_mat, negative_att_mat), \
        (template_schemas, positive_schemas, negative_schemas), (template_schema_idx, positive_schema_idx, negative_schema_idx), \
        (prior_temp_pos_mat, prior_temp_pos_mask_mat), (prior_temp_neg_mat, prior_temp_neg_mask_mat) = batch_data