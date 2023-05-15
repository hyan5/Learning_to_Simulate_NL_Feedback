from collections import Counter
from os import stat

from config import KEEP_TABLE_NAME
from config import REPLACE_UNDERSCORE_WITH_SPACE
from config import KEEP_VALUE
from config import SPIDER_TABLES
from config import USE_MODIFIED_SCHEMA


import selection_rules as sr

import re

import json

from typing import List
from typing import Dict
import pdb

import networkx as nx

AGG_OPS_LIN = ('', 'maximum', 'minimum', 'number of', 'summation of', 'average')
UNIT_OPS_LIN = ('', 'minus', 'plus', 'times', 'divided by')

KEY_PHRASE_MAP = {
    str(['stuid']): [['stuid'], ['student', 'id'], ['stu', 'id']],
    str(['distinct']): [["distinct"], ["distinctive"], ["distinctively"], ["repeat"], ["repeated"], ["repetition"], ["repetitious"], ["duplicate"], ['different']],
    str(['countrycode']): [['countrycode'], ['code'], ['that', 'country']],
    str(['charge', 'type']): [['charge', 'type'], ['charger', 'type']],
    str(['student', 'id']): [['student', 'id'], ['id', 'highschooler', 'table']],
    str(['other', 'student', 'details']):[['other', 'student', 'details'], ['other', 'pupils', 'information']],
    str(['loser', 'name']): [['loser', 'name'], ['name', 'winner', 'and', 'loser']],
    str(['governmentform']):[['governmentform'], ['government', 'forms']],
    str(['course', 'description']): [['course', 'description'], ['course', 'explanation']],
    str(['winner', 'name']): [['winner', 'name'], ['winners', 'wta', 'championship'], ['wta', 'championships']],
    str(['template', 'id']): [['template', 'id'], ['template', 'type', 'code']],
}

# Load json file
def load_json(path: str) -> List[Dict]:
    jsons = None
    with open(path, 'r') as f:
        jsons = f.read()
    
    jsons = json.loads(jsons)

    return jsons

# # get the BPE encoder
# CODER = None
# def get_bpe() -> Encoder:
#     global CODER
#     if CODER is None:
#         CODER = get_encoder(ENCODER, VOCAB)

#     return CODER

# # tokenize a txt string using the BPE encoder
# def tokenize(txt: str) -> list:
#     encoder = get_bpe()
#     codes = encoder.encode(txt)
#     tokens = []
#     for code in codes:
#         tokens += [encoder.decode([code])]

#     return tokens

# # tokenize a txt string list using the BPE encoder
# def tokenizeList(txt: list) -> list:
#     tokenized = []
#     for word in txt:
#         tokenized += tokenize(word)

#     return tokenized

# using a slide window to match a keyPhrase token list to a txt string
def slideWindowMatch(keyPhrase: list, txt: str) -> bool:
    txt = txt.replace('_', ' ').replace('-', ' ').replace('"', ' ').replace("'", ' ').replace('.', ' ').strip().lower().split()
    txt = tokenizeList(txt)
    txt_len = len(txt)
    keyPhrase = tokenizeList(keyPhrase)
    key_len = len(keyPhrase)
    if key_len > txt_len:
        return False

    # using a window size of 1 + txt_len
    padding = txt_len - key_len
    if padding > 2:
        padding = 2

    for i in range(txt_len - key_len - padding + 1):
        window = txt[i:i + key_len + padding]
        # print(window)

        match = True
        for token in keyPhrase:
            match = match * (token in window)   # make sure all tokens appeared in the window

        if match:
            return True

    return False

# match a keyphrase set to a txt string
def keyPhraseSetMatch(keyPhrase: list, txt: str) -> bool:
    # check the keyPhrase set
    if str(keyPhrase) in KEY_PHRASE_MAP:
        keyPhrases = KEY_PHRASE_MAP[str(keyPhrase)]

        for keyPhrase in keyPhrases:
            if slideWindowMatch(keyPhrase, txt):
                return True
        
        return False

    return slideWindowMatch(keyPhrase, txt)

class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema, table):
        self._schema = schema
        self._table = table
        self._idMap = self._map(self._schema, self._table)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema, table):
        column_names_original = table['column_names_original']
        table_names_original = table['table_names_original']
        #print 'column_names_original: ', column_names_original
        #print 'table_names_original: ', table_names_original
        for i, (tab_id, col) in enumerate(column_names_original):
            if tab_id == -1:
                idMap = {'*': i}
            else:
                key = table_names_original[tab_id].lower()
                val = col.lower()
                idMap[key + "." + val] = i

        for i, tab in enumerate(table_names_original):
            key = tab.lower()
            idMap[key] = i

        return idMap

_data = None
def get_schemas_from_json(fpath):
    global _data
    if _data is None:
        with open(fpath) as f:
            data = json.load(f)
            _data = data
    else:
        data = _data
    db_names = [db['db_id'] for db in data]

    tables = {}
    schemas = {}
    for db in data:
        db_id = db['db_id']
        schema = {} #{'table': [col.lower, ..., ]} * -> __all__
        column_names_original = db['column_names_original']
        table_names_original = db['table_names_original']
        tables[db_id] = {'column_names_original': column_names_original, 'table_names_original': table_names_original}
        for i, tabn in enumerate(table_names_original):
            table = str(tabn.lower())
            cols = [str(col.lower()) for td, col in column_names_original if td == i]
            schema[table] = cols
        schemas[db_id] = schema

    return schemas, db_names, tables

class Mappings(Schema):
    origin_schema = load_json(SPIDER_TABLES)
    db_schemas = None

    def __init__(self, schema: Schema, db_id: str):
        super().__init__(schema._schema, schema._table)

        self.db_id = db_id

        if Mappings.db_schemas == None:
            db_schemas = dict()
            for tab in Mappings.origin_schema:
                db_id = tab['db_id']
                tab_low2origin = dict([[t.lower(), t] for t in tab['table_names_original']])
                tab_low2name = dict([[tab_low, name] for tab_low, name in zip(list(tab_low2origin.keys()), list(tab['table_names']))])
                tab_name2low = dict((v, k) for k, v in tab_low2name.items())
                tab_name2origin = dict([[name, t] for name, t in zip(list(tab_name2low.keys()), list(tab['table_names_original']))])

                col_low2origin = dict([[f"{tab['table_names_original'][c[0]].lower()}.{c[1].lower()}", f"{tab['table_names_original'][c[0]]}.{c[1]}"] for c in tab['column_names_original'][1:]])
                col_low2name = dict([[col_low, name] for col_low, name in zip(list(col_low2origin.keys()), list(map(lambda x: x[1], tab['column_names'][1:])))])
                nodes = list(col_low2origin.keys())
                graph = nx.DiGraph()
                graph.add_nodes_from(nodes)
                for pair in tab['foreign_keys']:
                    graph.add_edge(nodes[pair[0] - 1], nodes[pair[1] - 1])
                graph = graph.to_undirected()
                sub_graphs = nx.connected_components(graph)
                graph_set = []
                for sub_graph in sub_graphs:
                    graph_set.append(sub_graph)
                sub_graphs = graph_set

                foreign_keys = dict()
                for k in nodes:
                    for sub_graph in sub_graphs:
                        if k in sub_graph:
                            foreign_keys[k] = sub_graph
                            break

                db_schemas[db_id] = {'tab_low2origin': tab_low2origin, 'tab_low2name': tab_low2name, 'tab_low2name': tab_low2name, 'tab_name2origin': tab_name2origin, 'col_low2origin': col_low2origin, 'foreign_keys': foreign_keys, 'col_low2name': col_low2name}

            Mappings.db_schemas = db_schemas

        self.inverseIdMap()

        return

    def inverseIdMap(self):
        columns = []
        tables = []
        targets = [columns, tables]

        previous = -1
        pointer = 0
        for item in self._idMap.items():
            current_name = item[0]
            current_id = item[1]

            if current_id <= previous:
                pointer = 1
            
            targets[pointer].append((current_id, current_name))
            previous = current_id

        self._id2Col = dict(targets[0])
        self._id2Tab = dict(targets[1])
        return

    @property
    def id2Col(self) -> dict:
        return self._id2Col

    @property
    def id2Tab(self) -> dict:
        return self._id2Tab

    @property
    def col_low2origin(self) -> dict:
        return Mappings.db_schemas[self.db_id]['col_low2origin']

    @property
    def col_low2name(self) -> dict:
        return Mappings.db_schemas[self.db_id]['col_low2name']

    @property
    def tab_low2origin(self) -> dict:
        return Mappings.db_schemas[self.db_id]['tab_low2origin']

    @property
    def tab_low2name(self) -> dict:
        return Mappings.db_schemas[self.db_id]['tab_low2name']
    
    @property
    def tab_name2low(self) -> dict:
        return Mappings.db_schemas[self.db_id]['tab_name2low']
    
    @property
    def tab_name2origin(self) -> dict:
        return Mappings.db_schemas[self.db_id]['tab_name2origin']

    @property
    def foreign_keys(self) -> dict:
        return Mappings.db_schemas[self.db_id]['foreign_keys']

class SqlArguments:
    def __init__(self, args: dict = None) -> None:
        self.values = {}

        if args is not None:
            self.args = args
        else:
            self.args = {}
            self.args['select'] = []
            self.args['from'] = []
            self.args['where'] = []
            self.args['groupBy'] = []
            self.args['orderBy'] = []
            self.args['having'] = []
            self.args['limit'] = []
            self.args['intersect'] = []
            self.args['except'] = []
            self.args['union'] = []
            self.args['SUBS'] = []

        return

def setEdits(source: list, target: list) -> dict:
    source = Counter(source)
    target = Counter(target)

    adds = target - source
    rms = source - target

    edits = {}
    edits['adds'] = list(adds)
    edits['rms'] = list(rms)

    return edits

def find_subs(sentence: str) -> set:
    pattern = 'SUBS(\d+)'
    ids = set()

    sentence = sentence.split()
    for s in sentence:
        match = re.search(pattern, s)
        if match is not None:
            ids.add(int(match.group(1)) - 1)

    return ids

class EditParser:

    ruleWords = ['maximum', 'minimum', 'number', 'sum', 'average', 'minus', 'times', 'divided', 'by', 'plus', 'all', 'distinct', 'not', 'between', 'and', 'or', 'greater', 'less', 'than', 'to', 'equals', 'one', 'of', 'like', 'desc', 'asc', 'remove', 'add', 'descending', 'ascending']

    def __init__(self, mappings: Mappings) -> None:
        self.mappings = mappings
        self.subs = []
        self.values = {}

    def parse_col_unit(self, unit: tuple) -> str:
        try:
            assert isinstance(unit, tuple) and isinstance(unit[2], bool) and len(unit) == 3
        except:
            raise TypeError(f'col_unit type error: {str(unit)}')

        if unit[1] == 0:
            parsed = 'rows'
        else:
            col = self.mappings.id2Col[unit[1]]
            split = col.split('.')

            if not KEEP_TABLE_NAME:
                if USE_MODIFIED_SCHEMA:
                    col = self.mappings.col_low2name[col]
                else:
                    col = split[1]
            # if not KEEP_TABLE_NAME:
            #     col = col.split('.')[-1]
            parsed = col
        if unit[2]:
            parsed = 'distinct ' + parsed
        
        if unit[0] != 0:
            parsed = AGG_OPS_LIN[unit[0]] + ' ' + parsed

        return parsed

    def parse_val_unit(self, unit: tuple) -> str:
        assert isinstance(unit, tuple) and len(unit) == 3
        if unit[0] == 0:
            return self.parse_col_unit(unit[1])

        return self.parse_col_unit(unit[1]) + ' ' + UNIT_OPS_LIN[unit[0]] + ' ' + self.parse_col_unit(unit[2])

    def parse_table_unit(self, unit: tuple) -> str:
        assert isinstance(unit, tuple) and (unit[0] == 'sql' or unit[0] == 'table_unit')
        if unit[0] == 'sql':
            assert isinstance(unit[1], dict)
            self.subs.append(unit[1])
            idx = len(self.subs)
            return 'SUBS' + str(idx)
        
        if USE_MODIFIED_SCHEMA:
            return self.mappings.tab_low2name[self.mappings.id2Tab[unit[1]]]

        return self.mappings.id2Tab[unit[1]]

    def parse_val(self, val) -> str:
        assert val is not None
        if isinstance(val, str) or isinstance(val, int) or isinstance(val, float):
            if KEEP_VALUE:
                if isinstance(val, float):
                    if val % 1 == 0:
                        val = int(val)

                return str(val)
            else:
                return ''

        if isinstance(val, dict):
            self.subs.append(val)
            idx = len(self.subs)
            return 'SUBS' + str(idx)

        if isinstance(val, tuple):
            assert len(val) == 3
            if isinstance(val[1], int) and isinstance(val[2], bool):
                return self.parse_col_unit(val)
            return self.parse_val_unit(val)

    def parse_cond_unit(self, unit: tuple, keep_val_unit = True) -> str:
        assert len(unit) == 5
        not_op, op_id, val_unit, val1, val2 = unit
        parsed = self.parse_val(val_unit)

        if not keep_val_unit:
            parsed = ''

        if op_id == 1:
            if not_op:
                parsed += ' not between '
            else:
                parsed += ' between '
            parsed += self.parse_val(val1) + ' and ' + self.parse_val(val2)
        elif op_id == 2:
            if not_op:
                parsed += ' not equals ' + self.parse_val(val1)
            else:
                parsed += ' equals ' + self.parse_val(val1)
        elif op_id == 3:
            if not_op:
                parsed += ' not greater than ' + self.parse_val(val1)
            else:
                parsed += ' greater than ' + self.parse_val(val1)
        elif op_id == 4:
            if not_op:
                parsed += ' not less than ' + self.parse_val(val1)
            else:
                parsed += ' less than ' + self.parse_val(val1)
        elif op_id == 5:
            if not_op:
                parsed += ' not greater than or equals to ' + self.parse_val(val1)
            else:
                parsed += ' greater than or equals to ' + self.parse_val(val1)
        elif op_id == 6:
            if not_op:
                parsed += ' not less than or equals to ' + self.parse_val(val1)
            else:
                parsed += ' less than or equals to ' + self.parse_val(val1)
        elif op_id == 7:
            if not_op:
                parsed += ' not not equals to ' + self.parse_val(val1)
            else:
                parsed += ' not equals to ' + self.parse_val(val1)
        elif op_id == 8:
            if not_op:
                parsed += ' not one of ' + self.parse_val(val1)
            else:
                parsed += ' one of ' + self.parse_val(val1)
        elif op_id == 9:
            if not_op:
                parsed += ' not like ' + self.parse_val(val1)
            else:
                parsed += ' like ' + self.parse_val(val1)
        else:
            raise TypeError

        return parsed.strip()

    def parse_condition(self, condition: list) -> list:
        assert isinstance(condition, list) and len(condition) == 0 or len(condition) % 2 == 1

        args = []

        current_cond = ''
        for cond in condition:
            if isinstance(cond, str):
                if cond.lower() == 'and':
                    args.append(current_cond)
                    current_cond = ''
                elif cond.lower() == 'or':
                    current_cond += ' or '
                else:
                    raise TypeError
            else:
                current_cond += self.parse_cond_unit(cond)

        if len(current_cond) > 0:
            args.append(current_cond)

        return args

    def parse_select(self, select: tuple) -> list:
        assert isinstance(select, tuple)

        args = []

        if select[0]:
            args.append('distinct')

        for item in select[1]:
            if item[0] != 0:
                arg = AGG_OPS_LIN[item[0]] + ' ' + self.parse_val(item[1])
            else:
                arg = self.parse_val(item[1])
            args.append(arg)

        return args

    def parse_from(self, f: dict) -> list:
        assert isinstance(f, dict)

        args = []

        for unit in f['table_units']:
            arg = self.parse_table_unit(unit)
            args.append(arg)
        
        # conds = self.parse_condition(f['conds']) # Do not consider the ON condition in from

        return args

    def parse_where(self, where: list) -> list:
        assert isinstance(where, list)
    
        return self.parse_condition(where)

    def parse_groupBy(self, groupBy: list) -> list:
        assert isinstance(groupBy, list)

        args = []

        for unit in groupBy:
            args.append(self.parse_col_unit(unit))

        return args

    def parse_orderBy(self, orderBy) -> list:
        assert (isinstance(orderBy, list) and len(orderBy) == 0) or (isinstance(orderBy, tuple) and len(orderBy) == 2)

        args = []
        if isinstance(orderBy, list):
            return args

        for unit in orderBy[1]:
            arg = self.parse_val(unit)
            args.append(arg)

        if orderBy[0] == 'desc':
            args.append('descending')

        return args

    def parse_having(self, having: list) -> list:
        assert isinstance(having, list)

        return self.parse_condition(having)

    def parse_limit(self, limit) -> list:
        if limit is None:
            return []

        return [self.parse_val(limit)]

    def parse_intersect(self, intersect) -> list:
        if intersect is None:
            return []
        
        assert isinstance(intersect, dict)
        self.subs.append(intersect)
        idx = len(self.subs)
        return ['SUBS' + str(idx)]

    def parse_except(self, exc) -> list:
        if exc is None:
            return []

        assert isinstance(exc, dict)
        self.subs.append(exc)
        idx = len(self.subs)
        return ['SUBS' + str(idx)]

    def parse_union(self, union) -> list:
        if union is None:
            return []

        assert isinstance(union, dict)
        self.subs.append(union)
        idx = len(self.subs)
        return ['SUBS' + str(idx)]

    def parse_sql_tree(self, sql_tree: dict) -> SqlArguments:
        assert isinstance(sql_tree, dict)
        parsed = {}

        parsed['select'] = self.parse_select(sql_tree['select'])
        parsed['from'] = self.parse_from(sql_tree['from'])
        parsed['where'] = self.parse_where(sql_tree['where'])
        parsed['groupBy'] = self.parse_groupBy(sql_tree['groupBy'])
        parsed['orderBy'] = self.parse_orderBy(sql_tree['orderBy'])
        parsed['having'] = self.parse_having(sql_tree['having'])
        parsed['limit'] = self.parse_limit(sql_tree['limit'])
        parsed['intersect'] = self.parse_intersect(sql_tree['intersect'])
        parsed['except'] = self.parse_except(sql_tree['except'])
        parsed['union'] = self.parse_union(sql_tree['union'])

        parsed['SUBS'] = []
        for sub in self.subs:
            parser = EditParser(self.mappings)
            parsed['SUBS'].append(parser.parse_sql_tree(sub))
        self.subs = []

        return SqlArguments(parsed)

    def calEdits(self, source: SqlArguments, target: SqlArguments) -> dict:
        assert isinstance(source, SqlArguments) and isinstance(target, SqlArguments)
        edits = {}
        source = source.args
        target = target.args
        assert isinstance(source, dict) and isinstance(target, dict)
        edits['select'] = setEdits(source['select'], target['select'])
        edits['from'] = setEdits(source['from'], target['from'])
        edits['where'] = setEdits(source['where'], target['where'])
        edits['groupBy'] = setEdits(source['groupBy'], target['groupBy'])
        edits['orderBy'] = setEdits(source['orderBy'], target['orderBy'])
        edits['having'] = setEdits(source['having'], target['having'])
        edits['limit'] = setEdits(source['limit'], target['limit'])
        edits['intersect'] = setEdits(source['intersect'], target['intersect'])
        edits['except'] = setEdits(source['except'], target['except'])
        edits['union'] = setEdits(source['union'], target['union'])

        subs_len = max(len(source['SUBS']), len(target['SUBS']))
        edits['SUBS'] = []
        for i in range(subs_len):
            parser = EditParser(self.mappings)
            try:
                subSource = source['SUBS'][i]
            except IndexError:
                subSource = SqlArguments()
            # finally:
            #     assert isinstance(subSource, SqlArguments)
            try:
                subTarget = target['SUBS'][i]
            except IndexError:
                subTarget = SqlArguments()
            # finally:
            #     assert isinstance(subTarget, SqlArguments)
            
            subEdit = parser.calEdits(subSource, subTarget)
            edits['SUBS'].append(subEdit)

        # Prune the removed SUBS
        sub_ids = set(range(subs_len))
        sub_rms = set()
        sub_adds = set()
        for clause in edits.items():
            k, v = clause
            if k == 'SUBS':
                continue

            adds = str(v['adds'])
            rms = str(v['rms'])
            adds = find_subs(adds)
            rms = find_subs(rms)

            sub_rms = set.union(sub_rms, rms)
            sub_adds = set.union(sub_adds, adds)

        sub_ids = sub_ids - sub_rms
        sub_ids = set.union(sub_ids, sub_adds)

        pruned = []
        ids = list(sub_ids)
        ids.sort()

        for id in ids:
            pruned.append(edits['SUBS'][id])
        edits['SUBS'] = pruned

        return edits 

    @staticmethod
    def linearize(edits: dict, sub_id: str = None, includeRm: bool = True) -> str:
        sentence = ''

        schema_info = []
        for k, v in edits.items():
            if k == 'SUBS':
                continue
            
            if len(v['adds']) + len(v['rms']) == 0:
                continue

            # sentence += f'<{k}> '
            tagIn = f'< {k} > '
            tagOut = f'</ {k} > '
            for arg in v['adds']:
                assert isinstance(arg, str)
                tag = tagIn + f'add {arg} ' + tagOut
                sentence += tag
                schema_info.append(arg)

            for arg in v['rms']:
                assert isinstance(arg, str)
                if not includeRm:
                    break
                tag = tagIn + f'remove {arg} ' + tagOut
                schema_info.append(arg)

                if sub_id is not None:
                    tag = f'{tag} '
                sentence += tag
        subs = edits['SUBS']
        assert isinstance(subs, list)
        for id, sub in enumerate(subs):
            next_sub_id = f'SUBS{str(id + 1)}'
            s, schema = EditParser.linearize(sub, next_sub_id, includeRm)
            # s = f'< SUBS{str(id+1)} > {s} </ SUBS{str(id+1)} >'
            schema_info += schema
            sentence += s
            # TODO: fix the potential SUBS index error

        # if sub_id is not None:
        #     sentence = f'<{sub_id}> {sentence}</{sub_id}> '

        if not KEEP_TABLE_NAME:
            pattern = r'[a-zA-Z_]+\.'
            sentence = re.sub(pattern, '', sentence)
            # parts = sentence.split()
            # sentence = ''
            # for part in parts:
            #     part = part.split('.')[-1]
            #     sentence += part + ' '

        if REPLACE_UNDERSCORE_WITH_SPACE:
            sentence = sentence.replace('_', ' ')
        return sentence.strip(), schema_info

    @staticmethod
    def editSize(edits: dict) -> int:
        size = 0
        for k, v in edits.items():
            if k == 'SUBS':
                continue
            
            size += len(v['adds'])
            size += len(v['rms'])

        for sub in edits['SUBS']:
            size += EditParser.editSize(sub)

        return size

    @staticmethod
    def findKeyPhrase(edits: dict) -> list:
        keyPhrases = []
        subPhrases = []

        subs = edits['SUBS']
        assert isinstance(subs, list)
        for sub in subs:
            subPhrases += EditParser.findKeyPhrase(sub)

        # for k, v in edits.items():
        #     if k != 'select':
        #         continue
        #     if k == 'select' or k == 'from':
        #         args = v['adds']
        #     else:
        #         args = v['adds'] + v['rms']
            
        #     for arg in args:
        #         phrases = arg.strip().split()
        #         for phrase in phrases:
        #             if (not phrase in EditParser.ruleWords) and (not phrase.lower().startswith('subs')):

        #                 phrase = phrase.replace('_', ' ').replace('-', ' ').replace('"', ' ').replace("'", ' ').replace(',', ' ').lower().split()
                        
        #                 # codes = []
        #                 # for token in phrase:
        #                 #     codes += coder.encode(token)

        #                 # phrase = list(map(lambda x: coder.decode([x]), codes))

        #                 keyPhrases += [phrase]

        keyPhrases =  sr.rule1(edits) + sr.rule2(edits) + sr.rule3(edits) + sr.rule4(edits) + sr.rule5(edits) + sr.rule7(edits) + \
            sr.rule8(edits) + sr.rule9(edits) + sr.rule11(edits) + sr.rule12(edits)


        return keyPhrases + subPhrases

    
                        
    @staticmethod
    def cal_coverage(keyPhrases: list, txt: str) -> tuple:
        pattern = r'[a-zA-Z_]+\.'
        txt = txt.strip()
        txt = txt.strip('.')
        txt = re.sub(pattern, ' ', txt)
        txt = txt.replace('_', ' ').replace('-', ' ').replace('"', ' ').replace("'", ' ').lower().split()

        # codes = []
        # for token in txt:
        #     codes += coder.encode(token)
        # txt = list(map(lambda x: coder.decode([x]), codes))

        nPhrase = len(keyPhrases)
        nTxt = len(txt)
        match = 0

        for phrase in keyPhrases:
            length = len(phrase)
            
            if length > nTxt:
                continue

            if length <= nTxt - 1:
                pad = 1
            else:
                pad = 0

            for i in range(nTxt - length + (1 - pad)):
                subtxt = txt[i: i + length + pad]
                matched = True
                for token in phrase:
                    matched = matched * (token in subtxt)

                if matched:
                    match += 1
                    break
            
            # if not matched:
            #     print(phrase)

        return match, nPhrase



# # %%
# from utils import load_json
# from utils import tokenize
# from utils import eval_hardness
# from config import SPIDER_TABLES
# from config import SPLASH_TRAIN_JSON
# from config import SPLASH_DEV_JSON
# from config import SPLASH_TEST_JSON
# from utils import parse_sql
# from collections import Counter
# from collections import deque
# from typing import List
# from typing import Tuple
# import json
# import random

# tables = load_json(SPIDER_TABLES)
# train = load_json(SPLASH_TRAIN_JSON)
# valid = load_json(SPLASH_DEV_JSON)
# test = load_json(SPLASH_TEST_JSON)

# with open('/home/yintao/projects/Interactive_Semantic_Parsing_Correction/data/splash/simulated/dev_qsw-td_low_sep_bart_large_best.sim', 'r') as f:
#     simFeedback = f.readlines()
# # data = train + valid + test
# data = test


# # %%
# SimUncovered = []
# FbUncovered = []
# numOfCases = 0
# editSizeExamples = [[], [], [], []]
# for id in range(len(data)):
#     sql1 = data[id]['gold_parse'].strip()
#     sql2 = data[id]['predicted_parse_with_values'].strip()
#     explanation = data[id]['predicted_parse_explanation']
#     db_id = data[id]['db_id']
#     question = data[id]['question']
#     feedback = data[id]['feedback']
#     sim = simFeedback[id]
#     sql_tree1, schema, table = parse_sql(sql1, db_id)
#     sql_tree2, schema, table = parse_sql(sql2, db_id)
#     mappings = Mappings(schema)
#     parser = EditParser(mappings)
#     target = parser.parse_sql_tree(sql_tree1)
#     source = parser.parse_sql_tree(sql_tree2)
#     edits = parser.calEdits(source, target)
#     edits_lin = parser.linearize(edits)
#     editSize = parser.editSize(edits)
#     keyPhrases = parser.findKeyPhrase(edits)
#     hardness = eval_hardness(sql_tree1)
#     feedbackCoverage = parser.cal_coverage(keyPhrases, feedback)
#     simCoverage = parser.cal_coverage(keyPhrases, sim)

#     if len(keyPhrases) > 0:
#         numOfCases += 1

#     for keyPhrase in keyPhrases:
#         if not keyPhraseSetMatch(keyPhrase, sim):
#             SimUncovered += [id]
#             break
#     for keyPhrase in keyPhrases:
#         if not keyPhraseSetMatch(keyPhrase, feedback):
#             FbUncovered += [id]
#             break

#     sample = {
#         'db_id': db_id,
#         'question': question,
#         'gold_parse': sql1,
#         'predicted_parse_with_values': sql2,
#         'explanation': explanation,
#         'feedback': feedback,
#         'simulation': sim,
#         'edits': edits_lin,
#         'keyPhrases': keyPhrases,
#         'editSize': editSize,
#         'hardness': hardness,
#         'feedback_coverage': feedbackCoverage,
#         'sim_coverage': simCoverage
#     }

#     if editSize <= 2:
#         editSizeExamples[0] += [sample]
#     elif editSize <= 4:
#         editSizeExamples[1] += [sample]
#     elif editSize <= 6:
#         editSizeExamples[2] += [sample]
#     else:
#         editSizeExamples[3] += [sample]

# print(len(SimUncovered))
# print(len(FbUncovered))
# print(numOfCases)

# # %%
# # print(SimUncovered)
# print(FbUncovered)
# # print(SimUncovered == FbUncovered)
# # %%
# id = 47
# db_id = data[id]['db_id']
# question = data[id]['question']
# sql1 = data[id]['gold_parse'].strip()
# sql2 = data[id]['predicted_parse_with_values'].strip()
# db_id = data[id]['db_id']
# feedback = data[id]['feedback']
# sim = simFeedback[id]
# explanation = data[id]['predicted_parse_explanation']
# sql_tree1, schema, table = parse_sql(sql1, db_id)
# sql_tree2, schema, table = parse_sql(sql2, db_id)
# mappings = Mappings(schema)
# parser = EditParser(mappings)
# target = parser.parse_sql_tree(sql_tree1)
# source = parser.parse_sql_tree(sql_tree2)
# edits = parser.calEdits(source, target)
# edits_lin = parser.linearize(edits)
# keyPhrases = parser.findKeyPhrase(edits)
# keyPhraseSetMatch(keyPhrases[0], feedback)

# print(db_id)
# print(question)
# print(sql1)
# print(sql2)
# print(explanation)
# print(edits_lin)
# print(keyPhrases)
# print(feedback)
# print(sim)

# # %%
# with open('1-2EditSizeExamples.txt', 'w') as f:
#     f.write(json.dumps(random.sample(editSizeExamples[0], 10), indent=4))
# with open('2-4EditSizeExamples.txt', 'w') as f:
#     f.write(json.dumps(random.sample(editSizeExamples[1], 10), indent=4))
# with open('4-6EditSizeExamples.txt', 'w') as f:
#     f.write(json.dumps(random.sample(editSizeExamples[2], 10), indent=4))
# with open('7+EditSizeExamples.txt', 'w') as f:
#     f.write(json.dumps(random.sample(editSizeExamples[3], 10), indent=4))
# # %%

# %%
