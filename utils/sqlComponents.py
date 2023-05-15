from parse_edit import Mappings

from config import KEEP_TABLE_NAME, USE_MODIFIED_SCHEMA, LOWER_SCHEMA_NAME
import config
from util_tools import parse_sql

import re
import pdb

AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
AGG_UTTR = {'none': '', 'max': 'maximum', 'min': 'minimum', 'count': 'number of', 'sum': 'summation of', 'avg': 'average'}
UNIT_OPS = ('none', '-', '+', "*", '/')
OPS_UTTR = {'none': '', '-': 'minus', '+': 'plus', "*": 'times', '/': 'divided by'}
COND_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
COND_UTTR = {'not': 'not', 'between': 'between', '=': 'equals', '>': 'greater than', '<': 'less than', '>=': 'greater than or equals', '<=': 'less than or equals', '!=': 'not equals', 'in': 'one of', 'like': 'like', 'is': '#undefined#', 'exists': '#undefined#'}


# %%
class SQLComponent(object):
    # Here are some SQL context variables
    mappings = None
    db_id = None
    schema = None
    table = None
    tables = []
    current_sub_quiries = []
    in_cond = False

    def __init__(self, is_empty = False) -> None:
        super().__init__()
        self.is_empty = is_empty
        self.is_nested = False
        self.sub_quiries = []

    def __str__(self) -> str:
        return '#undefined#'

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __eq__(self, o: object) -> bool:
        return hash(self) == hash(o)
    
    def added(self) -> str:
        if not config.SHOW_TAG:
            return str(self)
        return str(self)

    def subtracted(self) -> str:
        if not config.SHOW_TAG:
            return str(self)
        return str(self)

    def info(self) -> str:
        if not config.SHOW_TAG:
            return str(self)
        return str(self)


    def simplify(self) -> object:
        # some structures might be simplified into another simpler structure
        return self

    def component_type(self) -> str:
        return '#undefined#'

    def empty(self) -> None:
        # set the component as an empty component
        self.is_empty = True

class ColUnit(SQLComponent):
    related_tables = None
    single_sql_context = False
    correct_join_on_pairs = None
    # A column unit (agg_id, col_id, isDistinct(bool))
    def __init__(self, col_unit: dict) -> None:
        super().__init__()
        self.col_unit = col_unit
        self.name = SQLComponent.mappings.id2Col[col_unit[1]]
        self.agg = AGG_OPS[col_unit[0]]
        self.distinct = col_unit[2]
        self.schema = SQLComponent.schema
        self.tables = SQLComponent.tables

        if self.name == '*':
            self.foreign_keys = set()
        else:
            self.foreign_keys = SQLComponent.mappings.foreign_keys[self.name]

        self.col_low2origin = SQLComponent.mappings.col_low2origin
        self.col_low2name = SQLComponent.mappings.col_low2name
        self.tab_low2name = SQLComponent.mappings.tab_low2name
        self.tab_name2origin =SQLComponent.mappings.tab_name2origin

        if self.name == '*':
            self.col_name = 'rows'
            self.table = ''
        else:
            self.col_name = self.name.split('.')[1]
            self.table = self.name.split('.')[0]

        self.foreign_key_hash = sum(list(map(lambda x: hash(x), self.foreign_keys)))

    def __str__(self) -> str:
        name  = self.name
        if name == '*':     # replace "*" with "rows"
            column_name = 'rows'
            modified_col = column_name
            table_name = ''
        else:
            modified_col = self.col_low2name[name]

            if not LOWER_SCHEMA_NAME:
                name = self.col_low2origin[name]
            split = name.split('.')
            column_name = split[1]

            if USE_MODIFIED_SCHEMA:
                table_name = self.tab_low2name[split[0].lower()]
            else:
                table_name = split[0]

        uttr = ''
        if self.agg != 'none':
            uttr += AGG_UTTR[self.agg] + ' '

        if self.distinct:
            uttr += 'different '
        
        if not KEEP_TABLE_NAME:
            # Remove the table name
            other_columns = []
            if ColUnit.related_tables is not None:
                related_tables = ColUnit.related_tables
            else:
                related_tables = self.tables
            if ColUnit.single_sql_context:
                related_tables = self.tables
            
            for tab in related_tables:
                if tab != table_name.lower() and not isinstance(tab, Query):
                    if USE_MODIFIED_SCHEMA:
                        other_columns += self.schema.schema[self.tab_name2origin[tab].lower()]
                    else:
                        other_columns += self.schema.schema[tab]

            # if column_name in other_columns and not SQLComponent.in_cond:
            if column_name.lower() in other_columns:
                if USE_MODIFIED_SCHEMA:
                    uttr += table_name + '\'s ' + modified_col
                else:
                    uttr += table_name + '\'s ' + column_name
            else:
                if USE_MODIFIED_SCHEMA:
                    uttr += modified_col
                else:
                    uttr += column_name
        else:
            if USE_MODIFIED_SCHEMA:
                uttr += table_name + '\'s ' + modified_col
            else:
                uttr += table_name + '\'s ' + column_name

        if config.REPLACE_UNDERSCORE_WITH_SPACE:
            uttr = uttr.replace('_', ' ')
        
        return uttr
    
    def __hash__(self) -> int:
        if config.ONLY_COLUMN_NAME:
            return hash(self.col_name) + hash(self.agg) + hash(self.distinct)
        if not config.CONNECT_FOREIGN_KEY_GROUP:
            return hash(self.name) + hash(self.agg) + hash(self.distinct)
        name_hash = self.foreign_key_hash
        if ColUnit.correct_join_on_pairs is not None and len(ColUnit.correct_join_on_pairs) > 0 and config.CONNECT_FOREIGN_KEY_GROUP_BY_JOIN_ON:
            pair = ColUnit.correct_join_on_pairs[0]
            if name_hash == pair[0] or name_hash == pair[1]:
                name_hash = pair[0] + pair[1]
        return name_hash + hash(self.agg) + hash(self.distinct)

    def __eq__(self, o: object) -> bool:
        return hash(self) == hash(o)

    def agg_uttr(self) -> str:
        return AGG_UTTR[self.agg]

    def component_type(self) -> str:
        return 'column_unit'


class OptUnit(SQLComponent):
    # An option unit, represents an option between two ColUnit (unit_op, col_unit1, col_unit2 / None)
    def __init__(self, val_unit: tuple) -> None:
        super().__init__()
        self.val_unit = val_unit
        self.col1 = ColUnit(val_unit[1])

        if not val_unit[2] is None:
            self.col2 = ColUnit(val_unit[2])
        else:
            self.col2 = None

        self.op = UNIT_OPS[val_unit[0]]

    def op_uttr(self) -> str:
        return OPS_UTTR[self.op]

    def agg_uttr(self) -> str:
        return AGG_UTTR[self.agg]

    def __str__(self) -> str:
        if self.col2 is None or self.op == 'none':
            return str(self.col1)
        
        return ' '.join([str(self.col1), OPS_UTTR[self.op], str(self.col2)])

    def simplify(self) -> SQLComponent:
        if self.op == 'none':   # There is no opetion in this OptUnit
            assert self.col2 is None, 'Unseen structure found!'
            return self.col1
        
        return self

    def component_type(self) -> str:
        return 'option_unit'


class Value(SQLComponent):
    # The value represents the content of a cell in the database, it could be int, float, or str
    def __init__(self, val) -> None:
        super().__init__()
        self.val = val

        if isinstance(self.val, str):
            self.val = val.replace('"', '').replace("'", '').strip()

    def __str__(self) -> str:
        if isinstance(self.val, float):
            if int(self.val) == self.val:
                return str(int(self.val))

        if isinstance(self.val, str):
            if config.VALUE_KEEP_QUOTE:
                return '"%s"' % self.val
            else:
                return self.val

        return str(self.val)

    def component_type(self) -> str:
        return "value"


class SelectArgu(SQLComponent):
    # An argument in the SELECT clause (agg_id, val_unit)
    def __init__(self, select_argu: tuple) -> None:
        super().__init__()
        self.select_argu = select_argu
        self.agg = AGG_OPS[select_argu[0]]
        self.val = OptUnit(select_argu[1])

    def __str__(self) -> str:
        if self.agg == 'none':
            return str(self.val)

        return ' '.join([AGG_UTTR[self.agg], str(self.val)])

    def agg_uttr(self) -> str:
        return AGG_UTTR[self.agg]

    def simplify(self) -> SQLComponent:
        val_simp = self.val.simplify()
        if self.agg == 'none':
            return val_simp
        if isinstance(val_simp, ColUnit):
            if val_simp.agg == 'none':
                val_simp.agg = self.agg
                return val_simp
        return self

    def component_type(self) -> str:
        return 'select_argument'


def parse_val(val) -> SQLComponent:
    # This function is used to parse ColUnit, OptUnit, SelectArgu, TableUnit, Query, and values, and choose the simplest structure for it
    if isinstance(val, str) or isinstance(val, int) or isinstance(val, float):
        return Value(val) # a value in SQL

    elif isinstance(val, tuple):  # Might be ColUnit, OptUnit or SelectArgu
        if len(val) == 3:       # Might be ColUnit or OptUnit
            if isinstance(val[0], int) and isinstance(val[1], int) and isinstance(val[2], bool):
                return ColUnit(val)
            elif isinstance(val[0], int) and isinstance(val[1], tuple) and (isinstance(val[2], tuple) or val[2] is None):
                return OptUnit(val).simplify()
            else:
                raise ValueError('Unmatched data structure found!')
        elif len(val) == 2:
            if isinstance(val[0], int) and isinstance(val[1], tuple):
                return SelectArgu(val).simplify()
            elif isinstance(val[1], int) and val[0] == 'table_unit':
                return TableUnit(val[1])
            elif val[0] == 'sql' and isinstance(val[1], dict):
                sub = Query(val[1])
                SQLComponent.current_sub_quiries += [sub]
                return sub
            else:
                raise ValueError('Unmatched data structure found!')
        else:
            raise ValueError('Unmatched data structure found!')
    elif isinstance(val, dict):    # an sub query
        return Query(val)
    else:
        raise ValueError('Unmatched data structure found!')


class TableUnit(SQLComponent):
    # The table unit structure. It do not has the nested SQL, be careful about it (table_type, table_id / sql)
    def __init__(self, table_id: int) -> None:
        super().__init__()
        self.table_id = table_id
        self.name = SQLComponent.mappings.id2Tab[table_id]
        self.tab_low2origin = SQLComponent.mappings.tab_low2origin
        self.tab_low2name = SQLComponent.mappings.tab_low2name

    def __str__(self) -> str:
        if USE_MODIFIED_SCHEMA:
            name = self.tab_low2name[self.name]
        else:
            if not LOWER_SCHEMA_NAME:
                name = self.tab_low2origin[self.name]
            else:
                name = self.name
        return name

    def __hash__(self) -> int:
        return hash(str(self).lower())

    def component_type(self) -> str:
        return 'table_unit'


class CondUnit(SQLComponent):
    # represents one condition / filter (not_op, op_id, val_unit, val1, val2)
    def __init__(self, cond_unit:tuple) -> None:
        super().__init__()
        self.cond_unit = cond_unit
        self.negation = cond_unit[0]
        self.op = COND_OPS[cond_unit[1]]
        self.val0 = OptUnit(cond_unit[2]).simplify()
        self.val1 = parse_val(cond_unit[3])
        if cond_unit[4] is None:
            self.val2 = None
        else:
            self.val2 = parse_val(cond_unit[4])

        for val in [self.val0, self.val1, self.val2]:
            # Check the nested SQLs
            if isinstance(val, Query):
                self.sub_quiries += [val]
                self.is_nested = True

        if self.negation:
            operation = 'not ' + COND_UTTR[self.op]
        else:
            operation = COND_UTTR[self.op]

        SQLComponent.in_cond = True
        if self.op == 'between':
            self.description = ' '.join([operation, str(self.val1), 'and', str(self.val2)])
        else:
            self.description = ' '.join([operation, str(self.val1)])
        SQLComponent.in_cond = False

    def op_uttr(self) -> str:
        return COND_UTTR[self.op]

    def __str__(self) -> str:
        SQLComponent.in_cond = True
        uttr = ' '.join([str(self.val0), self.description])
        SQLComponent.in_cond = False
        return uttr

    def component_type(self) -> str:
        return 'condition_unit'


class Select(SQLComponent):
    # the SELECT clause (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
    def __init__(self, select_clause: tuple) -> None:
        super().__init__()
        self.distinct = select_clause[0]

        self.args = [parse_val(x) for x in select_clause[1]]
    
    def __str__(self) -> str:
        if self.distinct:
            uttr = 'select distinct '
        else:
            uttr = 'select '

        return uttr + ', '.join([str(arg) for arg in self.args])

    def __hash__(self) -> int:
        hash_value = hash(self.distinct)
        for arg in self.args:
            hash_value += hash(arg)

        return hash_value

    def component_type(self) -> str:
        return 'select'


class From(SQLComponent):
    # the FROM clause
    def __init__(self, from_clause: tuple) -> None:
        super().__init__()
        self.tables = [parse_val(table) for table in from_clause['table_units']]
        SQLComponent.tables += filter(lambda x: isinstance(x, TableUnit), self.tables)
        self.conds = []
        for idx, cond in enumerate(from_clause['conds']):
            if idx % 2 == 1:
                continue
            self.conds += [CondUnit(cond)]

        self.join_on_pairs = []
        for cond in self.conds:
            if isinstance(cond.val0, ColUnit) and isinstance(cond.val1, ColUnit) and cond.op == '=':
                self.join_on_pairs += [(cond.val0.foreign_key_hash, cond.val1.foreign_key_hash)]

    def __str__(self) -> str:
        uttr = 'from ' + ', '.join([str(x) for x in self.tables])
        if len(self.conds) > 0:
            uttr += ' join on ' + ', '.join([str(x) for x in self.conds])
            
        return uttr

    def __hash__(self) -> int:
        hash_value = 0
        for table in self.tables:
            hash_value += hash(table)
        for cond in self.conds:
            hash_value += hash(cond)
        
        return hash_value

    def component_type(self) -> str:
        return 'from'


class Where(SQLComponent):
    # the WHERE clause 
    def __init__(self, where_clause: list) -> None:
        super().__init__()
        self.where_clause = where_clause
        self.conds = []
        self.connectors = []
        self.or_pairs = set()
        for idx, cond in enumerate(where_clause):
            if idx % 2 == 1:
                assert cond == 'and' or cond == 'or', 'incorrect connector in WHERE!'
                self.connectors += [cond]
            else:
                c = CondUnit(cond)
                self.conds += [c]
                if c.is_nested:
                    self.is_nested = True
        for idx, connector in enumerate(self.connectors):
            if connector == 'or':
                id1 = idx
                id2 = idx + 1
                self.or_pairs.add(hash(self.conds[id1]) + hash(self.conds[id2]))
        if len(self.conds) == 0:
            self.is_empty = True

    def __str__(self) -> str:
        if self.is_empty:
            return 'where'

        return 'where ' + ', '.join([str(x) for x in self.conds])

    def __hash__(self) -> int:
        hash_value = 0
        for cond in self.conds:
            hash_value += hash(cond)

        return hash_value

    def component_type(self) -> str:
        return 'where'


class GroupBy(SQLComponent):
    # the GROUPBY clause
    def __init__(self, groupby_clause: list) -> None:
        super().__init__()
        self.groupby_clause = groupby_clause
        self.args = [ColUnit(col) for col in groupby_clause]
        if len(self.args) == 0:
            self.is_empty = True

    def __str__(self) -> str:
        if self.is_empty:
            return 'groupBy'
        return 'groupBy ' + ', '.join([str(col) for col in self.args])

    def __hash__(self) -> int:
        return sum([hash(col) for col in self.args])

    def component_type(self) -> str:
        return 'groupBy'

        
class OrderBy(SQLComponent):
    # the ORDERBY clause
    def __init__(self, orderby_clause: tuple) -> None:
        super().__init__()
        self.orderby_clause = orderby_clause
        if len(orderby_clause) == 0:
            self.dir = ''
            self.args = []
            self.is_empty = True
            return
        
        self.dir = orderby_clause[0]
        assert isinstance(self.dir, str), "OrderBy direction is not string"
        self.args = [OptUnit(val).simplify() for val in orderby_clause[1]]

    def __hash__(self) -> int:
        if self.is_empty:
            return 0

        hash_value = hash(self.dir)
        hash_value += sum([hash(arg) for arg in self.args])

        return hash_value

    def __str__(self) -> str:
        uttr = 'orderby'
        if not self.is_empty:
            uttr += self.dir + ' ' + ', '.join([str(arg) for arg in self.args])

        return uttr

    def component_type(self) -> str:
        return 'orderBy'


class Having(SQLComponent):
    # the HAVING clause
    def __init__(self, having_clause: list) -> None:
        super().__init__()
        self.having_clause = having_clause
        self.conds = []

        for idx, cond in enumerate(having_clause):
            if idx % 2 == 1:
                continue    # TODO: consider the "or" "and"
            self.conds += [CondUnit(cond)]
        
        if len(self.conds) == 0:
            self.is_empty = True

    def __hash__(self) -> int:
        return sum([hash(cond) for cond in self.conds])

    def __str__(self) -> str:
        if self.is_empty:
            return 'having'
        return 'having ' + ', '.join([str(cond) for cond in self.conds])

    def component_type(self) -> str:
        return 'having'


class Limit(SQLComponent):
    # the LIMIT clause
    def __init__(self, limit_clause) -> None:
        super().__init__()
        self.limit_clause = limit_clause
        self.val = limit_clause
        if self.val is None:
            self.val = -1
            self.is_empty = True
    
    def __str__(self) -> int:
        if self.is_empty:
            return 'limit'
        return ' '.join(['limit', str(self.val)])

    def component_type(self) -> str:
        return 'limit'


class Intersect(SQLComponent):
    # the INTERSECT clause
    def __init__(self, intersect_clause) -> None:
        super().__init__()
        self.intersect_clause = intersect_clause

        if intersect_clause is not None:
            self.sub_query = Query(intersect_clause)
            return
        self.sub_query = None
        self.is_empty = True

    def __str__(self) -> str:
        if self.is_empty:
            return 'intersect'
        return 'intersect ' + str(self.sub_query)

    def __hash__(self) -> int:
        return hash(self.sub_query)

    def component_type(self) -> str:
        return 'intersect'


class Except(SQLComponent):
    # the EXCEPT clause
    def __init__(self, except_clause) -> None:
        super().__init__()
        self.except_clause = except_clause

        if except_clause is not None:
            self.sub_query = Query(except_clause)
            return
        self.sub_query = None
        self.is_empty = True

    def __str__(self) -> str:
        if self.is_empty:
            return 'except'
        return 'except ' + str(self.sub_query)

    def __hash__(self) -> int:
        return hash(self.sub_query)

    def component_type(self) -> str:
        return 'except'


class Union(SQLComponent):
    # the UNION clause
    def __init__(self, union_clause) -> None:
        super().__init__()
        self.union_clause = union_clause

        if union_clause is not None:
            self.sub_query = Query(union_clause)
            return
        self.sub_query = None
        self.is_empty = True

    def __str__(self) -> str:
        if self.is_empty:
            return 'union'
        return 'union ' + str(self.sub_query)

    def __hash__(self) -> int:
        return hash(self.sub_query)

    def component_type(self) -> str:
        return 'union'


class Query(SQLComponent):
    # the SQL query data structure
    def __init__(self, sql, db_id = None) -> None:
        # sql could be a parsed tree or a plain text SQL query, context could be a db_id or a mappings
        super().__init__()
        if isinstance(sql, dict) and db_id is None:
            self.sql_tree = sql
        else:
            assert isinstance(sql, str) and isinstance(db_id, str), "sql should be a plain text SQL query and context is the db_id"
            self.sql_tree, schema, table = parse_sql(sql, db_id)
            if SQLComponent.db_id != db_id:
                # update the SQL schemas
                SQLComponent.mappings = Mappings(schema, db_id)
                SQLComponent.schema = schema
                SQLComponent.db_id = db_id
                SQLComponent.table = table

        self.fromm = From(self.sql_tree['from'])    # Parse FROM before SELECT, thus the col_units can aware the repeated column name
        self.select = Select(self.sql_tree['select'])
        self.where = Where(self.sql_tree['where'])
        self.group_by = GroupBy(self.sql_tree['groupBy'])
        self.order_by = OrderBy(self.sql_tree['orderBy'])
        self.having = Having(self.sql_tree['having'])
        self.limit = Limit(self.sql_tree['limit'])
        self.intersect = Intersect(self.sql_tree['intersect'])
        self.exceptt = Except(self.sql_tree['except'])
        self.union = Union(self.sql_tree['union'])
        self.sub_query = None
        self.operation = None
        if not self.intersect.is_empty:
            self.sub_query = self.intersect.sub_query
            self.operation = 'intersect'
        elif not self.exceptt.is_empty:
            self.sub_query = self.exceptt.sub_query
            self.operation = 'except'
        elif not self.union.is_empty:
            self.sub_query = self.union.sub_query
            self.operation = 'union'

        self.clauses = [self.select, self.fromm, self.where, self.group_by, self.order_by, self.having, self.limit, self.intersect, self.exceptt, self.union]
        self.sub_quiries = SQLComponent.current_sub_quiries
        SQLComponent.current_sub_quiries = []
        SQLComponent.tables = []

    def __str__(self) -> str:
        return ' '.join([str(clause) for clause in self.clauses])

    def __hash__(self) -> int:
        return sum([hash(clause) for clause in self.clauses])

    def component_type(self) -> str:
        return "sql_query"

# %%
# The test cases
if __name__ == "__main__":
    from utils import load_json
    from utils import parse_sql

    from config import SPLASH_TRAIN_JSON, SPLASH_DEV_JSON, SPLASH_TEST_JSON

    from tqdm import tqdm

    data = load_json(SPLASH_TRAIN_JSON) + load_json(SPLASH_DEV_JSON) + load_json(SPLASH_TEST_JSON)

    # %%
    idx = 123
    db_id = data[idx]['db_id']
    gold_sql = data[idx]['gold_parse']
    pred_sql = data[idx]['predicted_parse_with_values']
    print(gold_sql)
    gold_query = Query(gold_sql, db_id)
    pred_query = Query(pred_sql, db_id)

    print(gold_query.select.args[0])
    print(gold_query)
    print(pred_query)
    print(gold_query == pred_query)
    print(gold_query == gold_query)
    print(pred_query == pred_query)
    print(hash(gold_query))
    print(hash(pred_query))
    for a, b in zip(gold_query.clauses, pred_query.clauses):
        print(a.component_type(), a == b)

    # %%
    # test parsing all data
    for idx in tqdm(range(len(data))):
        db_id = data[idx]['db_id']
        gold_sql = data[idx]['gold_parse']
        pred_sql = data[idx]['predicted_parse_with_values']
        gold_query = Query(gold_sql, db_id)
        pred_query = Query(pred_sql, db_id)

    print('All data successfully parsed!')
    # %%
    # find subquries in WHERE CLAUSE
    num = 0
    for idx in tqdm(range(len(data))):
        db_id = data[idx]['db_id']
        gold_sql = data[idx]['gold_parse']
        pred_sql = data[idx]['predicted_parse_with_values']
        gold_query = Query(gold_sql, db_id)
        pred_query = Query(pred_sql, db_id)
        for cond in gold_query.where.conds:
            if isinstance(cond.val1, Query):
                num += 1
        for cond in pred_query.where.conds:
            if isinstance(cond.val1, Query):
                num += 1
    print(f"{num} / {len(data) * 2} sub SQL in WHERE clause")

    # %%
    # find subquries in FROM CLAUSE
    num = 0
    for idx in tqdm(range(len(data))):
        db_id = data[idx]['db_id']
        gold_sql = data[idx]['gold_parse']
        pred_sql = data[idx]['predicted_parse_with_values']
        gold_query = Query(gold_sql, db_id)
        pred_query = Query(pred_sql, db_id)
        for tab in gold_query.fromm.tables:
            if isinstance(tab, Query):
                num += 1
        for cond in pred_query.where.conds:
            if isinstance(tab, Query):
                num += 1
    print(f"{num} / {len(data) * 2} sub SQL in FROM clause")
    print('All data successfully parsed!')
