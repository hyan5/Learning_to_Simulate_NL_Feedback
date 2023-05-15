import json
from parse_sql_one import get_schemas_from_json
from parse_sql_one import Schema
from parse_sql_one import get_sql
from typing import List, Dict
from evaluation import count_component1
from evaluation import count_component2
from evaluation import count_others
from parse_edit import Mappings, SqlArguments
from parse_edit import EditParser
import pdb

from config import SPIDER_TABLES, USE_MODIFIED_SCHEMA

# The function to list schema
def list_schema(s: dict) -> str:
    if USE_MODIFIED_SCHEMA:
        columns = s['column_names']
    else:
        columns = s['column_names_original']
    schema = ''

    for c in columns[1:]:
        schema += c[1]
        schema += ' '

    return schema

# Process the sentence
def process_sentence(st: str, low: bool, sep: bool, dlm: str, app: str, strip: bool, rm: str = None) -> str:
    if strip:
        if rm is not None:
            st = st.strip(rm)
            st.strip()
        else:
            st = st.strip()
    if sep:
        st = dlm + st
    st = st + app
    if low:
        st = st.lower()

    return st

# Load json file
def load_json(path: str) -> List[Dict]:
    jsons = None
    with open(path, 'r') as f:
        jsons = f.read()
    
    jsons = json.loads(jsons)

    return jsons

def store_json(jsons: list, path: str) -> None:
    jsons = json.dumps(jsons, indent=2)
    with open(path, 'w') as f:
        f.write(jsons)
    return 

# Get the key from a splash sample
def get_splash_key(sample: Dict) -> int:
    key = 'db_id:' + sample['db_id'] + 'question:' + sample['question'] + 'gold_parse:' + sample['gold_parse'] + 'predicted_parse:' + sample['predicted_parse']
    key = hash(key)

    return key

# Parse the SQL into a tree
def parse_sql(sql: str, db_id: str) -> tuple:
    schemas, _, tables = get_schemas_from_json(SPIDER_TABLES)
    schema = schemas[db_id]
    table = tables[db_id]
    schema = Schema(schema, table)
    sql_label = get_sql(schema, sql)

    return sql_label, schema, tables

# Evaluate the hardness of a parsed gold SQL
def eval_hardness(sql: dict) -> str:
    count_comp1_ = count_component1(sql)
    count_comp2_ = count_component2(sql)
    count_others_ = count_others(sql)

    if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
        return "easy"
    elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
            (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
        return "medium"
    elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
            (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
            (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
        return "hard"
    else:
        return "extra"

# Calculate the Edits between two SQL queries
def cal_edits(source: str, target: str, db_id: str, includeRm: bool = True) -> str:
    # Parse the string SQL into a tree
    source, schema, table = parse_sql(source, db_id)
    target, schema, table = parse_sql(target, db_id)

    # Get the id to column/table name mappings
    mappings = Mappings(schema, db_id)
    parser = EditParser(mappings)

    # Parse the SQL tree into SQL arguments
    source = parser.parse_sql_tree(source)
    target = parser.parse_sql_tree(target)

    # Calculate the edits and linearize it
    edits = parser.calEdits(source, target)
    _edits = edits

    editSize = EditParser.editSize(edits)
    edits, schema_info = EditParser.linearize(edits, includeRm)

    return edits, editSize, schema_info, _edits

# Calculate the Edit Size between two SQL queries
def edit_size(source: str, target: str, db_id: str) -> int:
    # Parse the string SQL into a tree
    source, schema, table = parse_sql(source, db_id)
    target, schema, table = parse_sql(target, db_id)

    # Get the id to column/table name mappings
    mappings = Mappings(schema, db_id)
    parser = EditParser(mappings)

    # Parse the SQL tree into SQL arguments
    source = parser.parse_sql_tree(source)
    target = parser.parse_sql_tree(target)

    # Calculate the edits and the Edit Size
    edits = parser.calEdits(source, target)
    size = EditParser.editSize(edits)

    return size

# Calculate the feedback coverage of Edits of one feedback
def coverage(edits: dict, feedback: str) -> tuple:
    keyPhrases = EditParser.findKeyPhrase(edits)
    
    return EditParser.cal_coverage(keyPhrases, feedback)

def match_correctness(data: dict, p_str: str) -> bool:
    g_str = data['gold_parse']
    db = data['db_id']

# print the tree into a readable format
def printTree(tree: dict, indent: int = 4):
    if isinstance(tree, SqlArguments):
        tree = tree.args

    tree = json.dumps(tree, indent=indent)
    print(tree)

# print a dictionay into a readable format
def print_dict(data: dict) -> None:
    data = json.dumps(data, indent=2)
    print(data)