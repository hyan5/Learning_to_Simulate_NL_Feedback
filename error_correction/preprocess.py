import sys
import os
import random
import re
from config import SPLASH_DIR, SPLASH_TRAIN, SPLASH_DEV, SPLASH_TEST, SPIDER_TABLES, HOME, USE_MODIFIED_SCHEMA
from util_tools import list_schema, process_sentence, load_json, cal_edits, store_json
from templates import explanation
from os.path import join
from argparse import ArgumentParser
from nltk.tokenize import word_tokenize
from typing import Optional, List, Dict, Callable

from bridge_content_encoder import get_database_matches 

import pdb

parser = ArgumentParser(description="Preprocess the splash dataset into translation task format")
parser.add_argument("--low", action='store_true', help='Whether to convert corpus into lowercase', default=False)
parser.add_argument("--sep", action='store_true', help='Whether to add separation word', default=False)
parser.add_argument("--strip", action='store_true', help='Whether to strip the corpus', default=False)
parser.add_argument("--use_modified_schema", action='store_true', help='Whether to use the canical format of database schema', default=False)
parser.add_argument("--train", type=str, help='the training dataset path', required=True)
parser.add_argument("--dev", type=str, help='the development dataset path', required=True)
parser.add_argument("--test", type=str, help='the testing dataset path', required=True)
parser.add_argument("--format", type=str, help='the order how to contacanate each component', required=True)
parser.add_argument("--target", type=str, help='the target sentence. use "feedback" for human feedback simulator, use "edits" for error correction model.', required=True)
parser.add_argument("--out_dir", type=str, help='the output file path', required=True)
parser.add_argument("--use_template_feedback", action='store_true', help='Whether to use template feedback', default=False)

def prepare_data(args):
    args, _ = parser.parse_known_args(args)
    low = args.low
    sep = args.sep
    strip = args.strip
    out_dir = args.out_dir
    target = args.target
    use_modified_schema = args.use_modified_schema
    use_template = args.use_template_feedback

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_source_dir = join(out_dir, 'train.source')
    train_target_dir = join(out_dir, 'train.target')
    dev_source_dir = join(out_dir, 'dev.source')
    dev_target_dir = join(out_dir, 'dev.target')
    test_source_dir = join(out_dir, 'test.source')
    test_target_dir = join(out_dir, 'test.target')

    train_json = []
    dev_json = []
    test_json = []

    train_json = load_json(args.train)
    dev_json = load_json(args.dev)
    test_json = load_json(args.test)

        
    # Collect the feedbacks
    train_target = []
    dev_target = []
    test_target = []

    if target == 'gold_parse':
        for t in train_json:
            train_target.append(normalize(t[target]).replace("``", "\"").replace("''", "\""))

        for t in dev_json:
            dev_target.append(normalize(t[target]).replace("``", "\"").replace("''", "\""))

        if args.test:
            for t in test_json:
                test_target.append(normalize(t[target]).replace("``", "\"").replace("''", "\""))
    
    if target == 'feedback':
        for t in train_json:
            train_target.append(' '.join(word_tokenize(t[target])))
        for t in dev_json:
            dev_target.append(' '.join(word_tokenize(t[target])))
        if args.test:
            for t in test_json:
                test_target.append(' '.join(word_tokenize(t[target])))


    schema_json = load_json(SPIDER_TABLES)

    schema_json_dict = {}
    for s in schema_json:
        schema_json_dict[s['db_id']] = s

    schema_dict = {}

    global col_original2modified, tab_original2modified
    col_original2modified = {}
    tab_original2modified = {}

    for db_id, values in schema_json_dict.items():
        if db_id in schema_dict:
            continue
        tab_original = values['table_names_original']
        col_original = [col[1] for col in values['column_names_original'][1:]]
        tab_modified = values['table_names']
        col_modified = [col[1] for col in values['column_names'][1:]]

        col_original2modified[db_id] = dict(zip(col_original, col_modified))
        tab_original2modified[db_id] = dict(zip(tab_original, tab_modified))

        schema_dict[db_id] = {}
        tab_names = values['table_names_original']
        col_names = values['column_names_original']

        col_tab_id = []
        col_name = []

        for tab_id, name in col_names:
            col_tab_id.append(tab_id)
            col_name.append(name)
        schema_dict[db_id] = {'table_names': tab_names, 'column_names': {'table_id': col_tab_id, 'column_name': col_name}}

    # Write the targets to target file
    for dest_file, src_data in [(train_target_dir, train_target), (dev_target_dir, dev_target), (test_target_dir, test_target)]:
        if target == 'edits':
            continue
        with open(dest_file, 'w') as f:
            for t in src_data:
                _t1 = t
                if target == 'feedback':
                    t = process_sentence(' '.join(word_tokenize(t)), low, sep=False, dlm=None, app='\n', strip=strip, rm='')
                else:
                    t = process_sentence(t, low, sep=False, dlm=None, app='\n', strip=strip, rm='')
                _t2 = t
                f.write(t)

    # Write the input to the source files
    
    for dest_file, src_data in [(dev_source_dir, dev_json), (train_source_dir, train_json), (test_source_dir, test_json)]:
        with open(dest_file, 'w') as f:
            edits = []
            for i, t in enumerate(src_data):
                db_id = t['db_id']
                db_path = '../data/spider/database'
                db_table_names = schema_dict[db_id]['table_names']
                db_column_names = schema_dict[db_id]['column_names']
                question = ' '.join(word_tokenize(t['question'])).replace("``", "\"").replace("''", "\"")


                temp = t['template_feedback']
                gold_parse = normalize(t['gold_parse']).replace(" \"", " ``").replace("\"", " \"").replace("``", "\" ")
                predicted_parse = normalize(t['predicted_parse_with_values']).replace(" \"", " ``").replace("\"", " \"").replace("``", "\" ")

                schema_info = serialize_schema(question, db_path, db_id, db_column_names, db_table_names)

                q = process_sentence(question, low, sep, '[question] ', ' ', strip)
                s = process_sentence(schema_info, False, sep, '[schema] ', '', strip)
                c = process_sentence(gold_parse, low, sep, '[true] ', ' ', strip)
                w = process_sentence(predicted_parse, low, sep, '[predict] ', ' ', strip)
                exp = explanation(t['predicted_parse_with_values'], db_id)

                if target == 'edits':
                    feedback = ' '.join(word_tokenize(t['feedback'])).replace("``", "\"").replace("''", "\"")
                    fed = process_sentence(feedback, low, sep, '[feedback] ', ' ', strip)

                w_t = ''
                for wt in exp:
                    w_t += wt
                    w_t += '. '
                w_t = ' '.join(word_tokenize(w_t)).replace("``", "\"").replace("''", "\"")
                w_t = process_sentence(w_t, low, sep, '[system description] ', ' ', strip)


                d, edit_size, _edits_no_tag, _edits = cal_edits(t['predicted_parse_with_values'], t['gold_parse'], db_id)
                _d = ' '.join(word_tokenize(d)).replace("``", "\"").replace("''", "\"").replace("< /", "</")
                d = process_sentence(_d, low, sep, '', ' ', strip)
                temp = process_sentence(temp, low, sep, '', ' ', strip)
                edits.append(_d)
                source = ''
                for ch in args.format:
                    if ch == 'q':
                        source += q
                    if ch == 'f':
                        source += fed
                    if ch == 's':
                        source += s
                    if ch == 'c':
                        source += c
                    if ch == 'w':
                        source += w
                    if ch == 'e':
                        source += w_t
                    if ch == 'd':
                        source += d
                    if ch == 't':
                        source += temp
                source += '\n'
                _src = source
                # if 'train' in dest_file:
                #     pdb.set_trace()
                f.write(source)
                t['edits'] = _edits_no_tag
                t['edits_original'] = _d

            # if 'train' in dest_file:
            #     store_json(src_data, 'train.json')
            # if 'dev' in dest_file:
            #     store_json(src_data, 'dev.json')
            # if 'test' in dest_file:
            #     store_json(src_data, 'test.json')

            if target == 'edits':
                if 'train' in dest_file:
                    with open(train_target_dir, 'w') as f:
                        for edit in edits:
                            f.write('%s\n' %edit)
                if 'dev' in dest_file:
                    with open(dev_target_dir, 'w') as f:
                        for edit in edits:
                            f.write('%s\n' %edit)
                if 'test' in dest_file:
                    with open(test_target_dir, 'w') as f:
                        for edit in edits:
                            f.write('%s\n' %edit)

def serialize_schema(
    question: str,
    db_path: str,
    db_id: str,
    db_column_names: Dict[str, str],
    db_table_names: List[str],
    schema_serialization_type: str = "peteshaw",
    schema_serialization_randomized: bool = False,
    schema_serialization_with_db_id: bool = False,
    schema_serialization_with_db_content: bool = True,
    normalize_query: bool = True,
    use_modified_schema: bool = True,
) -> str:
    if schema_serialization_type == "verbose":
        db_id_str = "Database: {db_id}. "
        table_sep = ". "
        table_str = "Table: {table}. Columns: {columns}"
        column_sep = ", "
        column_str_with_values = "{column} ({values})"
        column_str_without_values = "{column}"
        value_sep = ", "
    elif schema_serialization_type == "peteshaw":
        db_id_str = " | {db_id}"
        table_sep = ""
        table_str = " [T] {table} [C] {columns}"
        column_sep = " [C] "

        column_str_with_values = "{column} ( {values} )"
        column_str_without_values = "{column}"
        value_sep = " , "
    else:
        raise NotImplementedError

    def get_column_str(table_name: str, column_name: str) -> str:
        if use_modified_schema:
            column_name_str = col_original2modified[db_id][column_name].lower() if normalize_query else col_original2modified[db_id][column_name]
        else:
            column_name_str = column_name.lower() if normalize_query else column_name

        if schema_serialization_with_db_content:
            matches = get_database_matches(
                question=question,
                table_name=table_name,
                column_name=column_name,
                db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
            )
            if matches:
                return column_str_with_values.format(column=column_name_str, values=value_sep.join(matches))
            else:
                return column_str_without_values.format(column=column_name_str)
        else:
            return column_str_without_values.format(column=column_name_str)

    tables = [
        table_str.format(
            table=tab_original2modified[db_id][table_name].lower() if use_modified_schema else table_name.lower(),
            columns=column_sep.join(
                map(
                    lambda y: get_column_str(table_name=table_name, column_name=y[1]),
                    filter(
                        lambda y: y[0] == table_id,
                        zip(
                            db_column_names["table_id"],
                            db_column_names["column_name"],
                        ),
                    ),
                )
            ),
        )
        for table_id, table_name in enumerate(db_table_names)
    ]
    if schema_serialization_randomized:
        random.shuffle(tables)
    if schema_serialization_with_db_id:
        serialized_schema = db_id_str.format(db_id=db_id) + table_sep.join(tables)
    else:
        serialized_schema = table_sep.join(tables)
    return serialized_schema

def normalize(query: str)->str:
    def comma_fix(s):
        return s.replace(" , ", ", ")
    def white_space_fix(s):
        return " ".join(s.split())
    def lower(s):
        return re.sub(r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s)
    return comma_fix(white_space_fix(lower(query)))

if __name__ == '__main__':
    args = sys.argv
    prepare_data(args)