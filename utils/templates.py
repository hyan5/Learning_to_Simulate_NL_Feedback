from pipes import Template
from sqlComponents import Query     # The convinent data structure for SQL queries
from sqlComponents import ColUnit, TableUnit, Value, OptUnit, SelectArgu, SQLComponent

import config
from config import RULES
from config import ABANDON
from config import ABANDONED
from config import ADD_TAG1
from config import ADD_TAG2
from config import SUB_TAG1
from config import SUB_TAG2
from config import INFO_TAG1
from config import INFO_TAG2
import traceback
from nltk.tokenize import word_tokenize
import pdb

def feedback(wrong_sql: str, correct_sql: str, db_id: str, verbal: bool=False) -> str:
    wrong_parse = Query(wrong_sql, db_id)
    correct_parse = Query(correct_sql, db_id)
    
    try:
        corrections, _ = get_feedback(wrong_parse, correct_parse)
        corrections = remove_step(corrections)
        correction = connect_sents(corrections)
        if not config.NO_NLTK_TOKENIZER:
            correction = ' '.join(word_tokenize(correction))

    except StructureError as e:
        # Only capture StructureError here, StructureError is defined sturctural error
        correction = "ERROR: " + repr(e)
        if verbal:
            print(f"Feedback generation failed for: [{db_id}] \n {wrong_sql} \n {correct_sql}")
            traceback.print_exc()
    
    return correction

def explanation(sql: str, db_id: str) -> str:
    parse = Query(sql, db_id)
    explanations, _ = get_explanation(parse)
    if not config.NO_NLTK_TOKENIZER:
        explanations = list(map(lambda x: ' '.join(word_tokenize(x)), explanations))
    
    return add_step(explanations)

def find_span(tag1, tag2, others, sentence) -> list:
    spans = []
    sentence = sentence.split()
    in_span = False
    current_span = [-1, -1]
    if config.NO_NLTK_TOKENIZER:
        for idx, word in enumerate(sentence):
            if tag1 in word and not in_span:
                current_span[0] = idx
                in_span = True
            if tag2 in word and in_span:
                current_span[1] = idx
                in_span = False
                spans.append(current_span)
                current_span = [-1, -1]
    else:
        tag1 = word_tokenize(tag1)
        tag2 = word_tokenize(tag2)
        others = list(map(lambda x: word_tokenize(x), others))
        len1 = len(tag1)
        len2 = len(tag2)
        true_idx = 0
        for idx in range(len(sentence) - len2 + 1):
            if tag1 == sentence[idx: idx + len1] and in_span:
                true_idx -= len1
            if tag1 == sentence[idx: idx + len1] and not in_span:
                current_span[0] = true_idx
                in_span = True
                true_idx -= len1
            if tag2 == sentence[idx: idx + len2] and not in_span:
                spans[-1][1] = true_idx - 1
                true_idx -= len2
            if tag2 == sentence[idx: idx + len2] and in_span:
                current_span[1] = true_idx - 1
                true_idx -= len2
                spans.append(current_span)
                current_span = [-1, -1]
                in_span = False
            for other in others:
                if other == sentence[idx: idx + len(other)]:
                    true_idx -= len(other)
            true_idx += 1

    return spans

class StructureError(Exception):     # The exception for unimplemented templates or abandoned templates
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def connect_words(words: list, append_word: str='', comma_only: bool=False, mode: str='') -> str:
    # connect_words [A, B, C, D] "aw" into "A aw, B aw, C aw, and D aw"
        words = list(words)
        units = []
        for word in words:
            if str(word):

                if not isinstance(word, SQLComponent):
                    units += [str(word)]
                elif mode == 'add':
                    units += [word.added()]
                elif mode == 'sub':
                    units += [word.subtracted()]
                elif mode == 'info':
                    units += [word.info()]
                else:
                    units += [str(word)]
        if len(units) == 0:
            return ''
        elif len(units) == 1:
            if append_word:
                return f"{units[0]} {append_word}"
            return units[0]
        elif len(units) == 2:
            if append_word:
                if comma_only: return f"{units[0]} {append_word}, {units[1]} {append_word}"
                return f"{units[0]} {append_word} and {units[1]} {append_word}"
            return f"{units[0]} and {units[1]}"

        sent = ''
        for word in units[0:-1]:
            if append_word:
                sent += f"{word} {append_word}, "
            else:
                sent += f"{word}, "
        
        if append_word:
            if comma_only: sent += f"{units[-1]} {append_word}"
            else: sent += f"and {units[-1]} {append_word}"
        else:
            if comma_only: sent += f"{units[-1]}"
            else: sent += f"and {units[-1]}"
        return sent

def connect_sents(sents: list, step: int=-1, rule: int=None):
    if rule is not None:
        assert isinstance(rule, int)
    # connect sentences, use a space to seperate them. If the first sentence is empty sentence, add "in step X, " at the begining.
    if sum(list(map(lambda x: len(x), sents))) == 0:
        return ''
    if not rule is None:
        if not RULES[rule]:
            if ABANDON:
                config.ABANDONED[rule] += 1
                raise StructureError(f'Error rule {rule} found!')
            return sents[0]     # Disable this rule
    sent = ''
    if sents[0] == '' and step > 0:
        sent += f"in step {str(step)}, "

    for s in sents:
        if s != '':
            if not config.NO_NLTK_TOKENIZER:    # because NLTK tokenizer will take "4." in "equals to 4." as one token. Thus add a space before the last full stop
                if s[-1] == '.':
                    s = s[:-1] + ' ' + '.'
            assert s[-1] != ' ', "space repeated!"
            if sent == '' or sent[-1] == ' ':
                sent += s
            else:
                sent += ' ' + s

    return sent

def add_step(explanation: list, step='Step') -> list:
    if len(explanation) == 1:
        return explanation
    added = []
    for idx, e in enumerate(explanation):
        added += [f'{step} %d: ' % (idx + 1) + e]

    return added

def remove_step(feedback: list) -> list:
    if len(feedback) == 1 and 'in step 1, ' in feedback[0]:
        feedback[0] = feedback[0][11:]
    return feedback    

def simple_explain(parse: Query) -> str:
    select = connect_words([str(x) for x in parse.select.args])
    if len(parse.fromm.tables) > 1: raise Exception("JOIN ON in the simple explain")
    table = str(parse.fromm.tables[0])
    groupby = ''
    if len(parse.group_by.args) > 0:
        groupby = f'of each value of {connect_words([str(x) for x in parse.group_by.args])}'
    where = ''
    if len(parse.where.conds) > 0:
        where = f"whose {connect_words([str(x) for x in parse.where.conds])}"
    orderby = ''
    if len(parse.order_by.args) > 0:
        orderby = f'ordered { {"asc": "ascending", "desc": "descending"}[parse.order_by.dir] } by {connect_words([str(x) for x in parse.order_by.args])}'
    if parse.limit.val == 1:
        limit = f'with {str({"asc": "smallest", "desc": "largest"}[parse.order_by.dir])} {connect_words([str(x) for x in parse.order_by.args])}'
        return connect_sents([select, table, groupby, where, limit])

    return connect_sents([select, table, groupby, where, orderby])

def add(uttr: str) -> str:
    if not config.SHOW_TAG:
        return str(uttr)
    if not config.NO_NLTK_TOKENIZER:
        return f"{ADD_TAG1} {uttr} {ADD_TAG2}"
    return f"{ADD_TAG1}{uttr}{ADD_TAG2}"

def sub(uttr: str) -> str:
    if not config.SHOW_TAG:
        return str(uttr)
    if not config.NO_NLTK_TOKENIZER:
        return f"{SUB_TAG1} {uttr} {SUB_TAG2}"
    
    return f"{SUB_TAG1}{uttr}{SUB_TAG2}"

def info(uttr: str) -> str:
    if not config.SHOW_TAG:
        return str(uttr)
    return f"{INFO_TAG1}{uttr}{INFO_TAG2}"

def feedback_context(func):
    # When judge the column name ambiguity in feedback generation, we should consider tables mentioned in all the SQL queries.
    def wrapper(*args, **kwargs):
        parameters = list(args) + list(kwargs.values())
        tables = []
        correct_join_on_pairs = []
        first = True
        for p in parameters:
            if isinstance(p, Query):
                if first:
                    first = False
                else:
                    correct_join_on_pairs = p.fromm.join_on_pairs
                tables += filter(lambda x: isinstance(x, TableUnit), p.fromm.tables)
        if ColUnit.related_tables is None:  # If the feedback is nested in a feedback invoke, we should not change the related_tables.
            nested = False
            ColUnit.related_tables = list(tables)
            ColUnit.correct_join_on_pairs = correct_join_on_pairs
        else:
            nested = True

        try:
            results = func(*args, **kwargs)     # Get the the results of the wrapped function.
        except Exception as exc:
            ColUnit.related_tables = None
            ColUnit.correct_join_on_pairs = None
            raise exc

        if not nested:
            ColUnit.related_tables = None
            ColUnit.correct_join_on_pairs = None
    
        return results
    return wrapper

def get_explanation(sql_parse: Query, step: int=1, join_table: bool=False):
    explanations = []
    tab_low2name = sql_parse.mappings.tab_low2name
    col_low2name = sql_parse.mappings.col_low2name

    if not sql_parse.intersect.is_empty:
        sub_sql = sql_parse.intersect.sub_query
        sql_parse.intersect.empty()
        sent1, step = get_explanation(sql_parse, step)
        part1_step = step - 1
        sent2, step = get_explanation(sub_sql, step)
        part2_step = step - 1
        sent3 = "show the rows that are in both the results of step %d and step %d" % (part1_step, part2_step)
        explanations = sent1 + sent2 + [sent3]
        step += 1

    elif not sql_parse.union.is_empty:
        sub_sql = sql_parse.union.sub_query
        sql_parse.union.empty()
        sent1, step = get_explanation(sql_parse, step)
        part1_step = step - 1
        sent2, step = get_explanation(sub_sql, step)
        part2_step = step - 1
        sent3 = "show the rows that are in any of the results of step %d and step %d" % (part1_step, part2_step)
        explanations = sent1 + sent2 + [sent3]
        step += 1

    elif not sql_parse.exceptt.is_empty:
        sub_sql = sql_parse.exceptt.sub_query
        sql_parse.exceptt.empty()
        sent1, step = get_explanation(sql_parse, step)
        part1_step = step - 1
        sent2, step = get_explanation(sub_sql, step)
        part2_step = step - 1
        sent3 = "show the rows that are in the results of step %d but not in the results of step %d" % (part1_step, part2_step)
        explanations = sent1 + sent2 + [sent3]
        step += 1
    
    elif not sql_parse.limit.is_empty and sql_parse.limit.val > 1:
        sent2 = "only show the first %d rows of the results" % sql_parse.limit.val
        # remove from sql_parse
        sql_parse.limit.empty()
        sent1, step = get_explanation(sql_parse, step)
        explanations = sent1 + [sent2]
        step += 1

    elif not sql_parse.fromm.is_empty and isinstance(sql_parse.fromm.tables[0], Query):
        # when nested sub query in FROM clause
        assert len(sql_parse.fromm.tables) == 1, "Apart of the subquery, more things in the FROM clause"
        sub_query = sql_parse.fromm.tables[0]
        sql_parse.fromm.empty()
        sent1, step = get_explanation(sub_query, step)
        sent2, step = get_explanation(sql_parse, step, join_table=True)

        explanations = sent1 + sent2
    
    elif not sql_parse.fromm.is_empty and len(sql_parse.fromm.tables) > 1: 
        # Assume no nested SQL in FROM in this case
        assert not isinstance(sql_parse.fromm.tables[0], Query), "subquery in non-single FROM clause"


        table_name1 = str(sql_parse.fromm.tables[0])
        sent1 = "for each row in %s table" % table_name1
        is_first = True
        for other_table_unit in sql_parse.fromm.tables[1:]:
            other_table_name = str(other_table_unit)
            if is_first:
                sent1 += ", find the corresponding rows in %s table" % other_table_name
                is_first = False
            else:
                sent1 += " and in %s table" % other_table_name
    
        # remove from sql_parse
        sql_parse.fromm.empty()
        sent2, step = get_explanation(sql_parse, step + 1, join_table=True)
        explanations = [sent1] + sent2
    
    else:
        bool_groupby = not sql_parse.group_by.is_empty
        bool_orderby = not sql_parse.order_by.is_empty
        bool_having = not sql_parse.having.is_empty
        bool_where = not sql_parse.where.is_empty
        
        if join_table:  # including join tables and nested SQL in FROM clause
            table_description = "of the results of step %d" % (step - 1)
        else:
            table_name = str(sql_parse.fromm.tables[0])
            table_description = "in %s table" % table_name


        if sql_parse.fromm.is_empty and not join_table:
            table_description = ''
        
        sel_column_names = [str(col) for col in sql_parse.select.args]
        
        # select ...
        if sql_parse.select.distinct:
            sel_sent = 'find without repetition %s' % connect_words(sel_column_names, comma_only=True)
        else:
            sel_sent = "find the %s" % connect_words(sel_column_names, comma_only=True)
        if sql_parse.select.is_empty:
            sel_sent = ''

        # the WHERE clause, there might be nested quiries within it, and the behavior depends on whether the GROUP BY exists
        def connect_where_conds(conds: list, connectors: list):
            # this function is used to connected WHERE conditions with correct "AND" "OR" connectors
            if len(conds) == 1:
                return str(conds[0])
            elif len(conds) == 0:
                raise Exception("No conditions in WHERE clause")
            uttr = ''
            for pair in zip(conds[:-1], connectors):
                uttr += f"{pair[0]} {pair[1]} "
            uttr += str(conds[-1])
            return uttr

        where_explanation = []  # for the nested SQL in WHERE clause
        where_description = ''
        if bool_where:
            where_description = "whose "
            where_descriptions = []
            for cond in sql_parse.where.conds:
                if cond.is_nested:
                    assert len(cond.sub_quiries) == 1, "more than one subquires in one cond_unit"
                    assert not isinstance(cond.val0, Query) and isinstance(cond.val1, Query), "new format of cond_unit founded!"
                    where_explanation, step = get_explanation(cond.sub_quiries[0], step)
                    where_descriptions += [connect_sents([str(cond.val0), 'not' if cond.negation else '', str(cond.op_uttr()), 'the results of step %d' % (step - 1)])]
                else:
                    where_descriptions +=[str(cond)]
            where_description += connect_where_conds(where_descriptions, sql_parse.where.connectors)
        
        if bool_groupby:
            groupby_column_names = [str(col) for col in sql_parse.group_by.args]

        if bool_orderby:
            orderby_column_names = [str(val_unit) for val_unit in sql_parse.order_by.args]
            orderby_direction = {"asc": "ascending", "desc": "descending"}[sql_parse.order_by.dir]
            orderby_est = {"asc": "smallest", "desc": "largest"}[sql_parse.order_by.dir]
            limit_value = sql_parse.limit.val
            if sql_parse.limit.is_empty:
                limit_value = None
            assert limit_value is None or limit_value == 1  # the LIMIT larger than 1 is processed outside
        
        if bool_having:
            assert len(sql_parse.having.conds) == 1, "assume only one cond_unit in HAVING"
            # assert not sql_parse.having.conds[0].is_nested, "no nested SQL in the HAVING clause"
            having_column_name = str(sql_parse.having.conds[0].val0)
            having_description = sql_parse.having.conds[0].description

        # This is for the case there are no GROUPBY
        # select ... from ... (where) ... (order by)
        if sql_parse.having.is_empty and sql_parse.group_by.is_empty:
            if sql_parse.fromm.is_empty and not join_table:
                table_description = ''
            sent1 = connect_sents([sel_sent, table_description])
            if bool_where:
                sent1 += ' ' + where_description
            if bool_orderby:
                if limit_value is None:
                    sent1 = connect_sents([sent1, 'ordered', orderby_direction, 'by', connect_words(orderby_column_names, comma_only=True)])
                else:
                    sent1 = connect_sents([sent1, 'with', orderby_est, 'value of', connect_words(orderby_column_names, comma_only=True)])
            
            return where_explanation + [sent1], step + 1

        # This is for the case where there are both WHERE and GROUPBY
        if bool_groupby and bool_where:
            assert len(groupby_column_names) == 1, "only one column name in groupby"
            if join_table:
                where_explanation += ["only keep the results of step %d %s" % (step - 1, where_description)]
            else:
                where_explanation += [connect_sents(["find rows", table_description, where_description])]
            table_description = "the results of step %d" % step
            if sql_parse.fromm.is_empty and not join_table:
                table_description = 'abc'
            step += 1

        # select ... (where ...) group by ...
        # e.g., SELECT Employee_ID , Count ( * ) FROM Employees GROUP BY Employee_ID ->
        # find each value of Employee_ID in Employees table along with the number of the corresponding rows to each value
        if bool_groupby and not (bool_orderby or bool_having):
            # assert not bool_where, "WHERE should not occur with GROUP BY"
            if "number of rows" in sel_column_names:
                sent = "find each value of %s %s along with the number of the corresponding rows to each value" %(connect_words(groupby_column_names), table_description)
            else:
                sent = "find each value of %s %s along with the %s of the corresponding rows to each value" % (connect_words(groupby_column_names),
                                                                                                                table_description,
                                                                                                                connect_words(list(set(sel_column_names) - set(groupby_column_names))))
            return where_explanation + [sent], step + 1
        
        # select ... (where ...) group by ... order by ...
        if bool_groupby and bool_orderby and not bool_having:
            # assert not bool_where, "WHERE should not occur with GROUP BY"
            sent1 = "find the %s of each value of %s %s" % (connect_words(orderby_column_names),
                                                        connect_words(groupby_column_names),
                                                        table_description)
            if limit_value and orderby_direction == "descending":
                sent2 = f"{sel_sent} {table_description} with largest value in the results of step {str(step)}"
            elif limit_value and orderby_direction == "ascending":
                sent2 = f"{sel_sent} {table_description} with smallest value in the results of step {str(step)}"
            else:
                sent2 = f"{sel_sent} {table_description} ordered {orderby_direction} by the results of step {str(step)}"

            return where_explanation + [sent1, sent2], step + 2
        
        # select ... (where ...) group by ... having ...
        if bool_groupby and bool_having and not bool_orderby:
            # assert not bool_where, "WHERE should not occur with GROUP BY"
            sent1 = "find the %s of each value of %s %s" % (having_column_name, 
                                                        connect_words(groupby_column_names),
                                                        table_description)
            sent2 = connect_sents([sel_sent, table_description, "whose corresponding value in step %d is %s" % (step, having_description)])

            return where_explanation + [sent1, sent2], step + 2

        # This case is only for feedback generation
        if bool_having and not bool_groupby:
            sent1 = 'make sure %s ' % connect_words(sql_parse.having.conds)
            
            return where_explanation + [sent1], step + 1
                    
        raise Exception("New patterns discovered!")
    return explanations, step


@feedback_context
def get_feedback(wrong_parse: Query, correct_parse: Query, step: int=1, join_table: bool=False):
    # if wrong_parse is not None and correct_parse is not None:
    #     if wrong_parse.where.is_nested != correct_parse.where.is_nested:
    #         if ABANDON:
    #             config.ABANDONED[101] += 1
    #         raise StructureError('Structural error')
    if wrong_parse.sub_query is None:   # add one SQL operation
        if correct_parse is not None:
            if correct_parse.sub_query is not None:
                extra_sub = correct_parse.sub_query
                correct_parse.sub_query = None
                sent1, step1 = get_feedback(wrong_parse, correct_parse, step)
                extra_exp, step2 = get_explanation(extra_sub)
                sent2 = add_step(extra_exp, step='extra step')
                sent3 = ''
                if correct_parse.operation == 'union':
                    sent3 = connect_sents([sent3, f"return the rows in {info('any of')} the results of step {step1 - 1} and the results of extra step {step2 - 1}."], rule=3)
                if correct_parse.operation == 'except':
                    sent3 = connect_sents([sent3, f"return the rows in the results of step {step1 - 1} {info('but not')} the results of extra step {step2 - 1}."], rule=3)
                if correct_parse.operation == 'intersect':
                    sent3 = connect_sents([sent3, f"return the rows in {info('both of')} the results of step {step1 - 1} and the results of extra step {step2 - 1}."], rule=3)

                if RULES[3]:
                    return sent1 + sent2 + [sent3], step1
                else:
                    return sent1, step1

    if not wrong_parse.intersect.is_empty:
        wrong_sub = wrong_parse.intersect.sub_query
        wrong_parse.intersect.empty()
        correct_sub = None
        if correct_parse is None:
            correct_sub = None
        elif correct_parse.sub_query is not None:
            correct_sub = correct_parse.sub_query
            correct_parse.sub_query = None
            correct_parse.intersect.empty()
            correct_parse.union.empty()
            correct_parse.exceptt.empty()

        sent1, step = get_feedback(wrong_parse, correct_parse, step)
        old_step = step
        sent2, step = get_feedback(wrong_sub, correct_sub, step)
        uttr = ''
        if correct_sub is None:
            empty_steps = []
            for _ in sent2:
                empty_steps += ['']
            sent2 = empty_steps
            if step - old_step > 1:
                uttr = connect_sents([uttr, 'remove step %d to step %d.' % (old_step, step)], rule=1)   # remove the sql operation
            else:
                uttr = connect_sents([uttr, 'remove step %d.' % step], rule=1)            
        elif correct_parse.operation == 'union':
            uttr = connect_sents([uttr, add(f"return the rows in {info('any of')} the results of step {old_step - 1} and the results of step {step - 1}.")], rule=2) # change the sql operator
        elif correct_parse.operation == 'except':
            uttr = connect_sents([uttr, add(f"return the rows in the results of step {old_step - 1} {info('but not')} the results of step {step - 1}.")], rule=2)

        feedbacks = sent1 + sent2 + [uttr]
        step += 1

    elif not wrong_parse.union.is_empty:
        wrong_sub = wrong_parse.union.sub_query
        wrong_parse.union.empty()
        correct_sub = None
        if correct_parse is None:
            correct_sub = None
        elif correct_parse.sub_query is not None:
            correct_sub = correct_parse.sub_query
            correct_parse.sub_query = None
            correct_parse.intersect.empty()
            correct_parse.union.empty()
            correct_parse.exceptt.empty()

        sent1, step = get_feedback(wrong_parse, correct_parse, step)
        old_step = step
        sent2, step = get_feedback(wrong_sub, correct_sub, step)
        uttr = ''
        if correct_sub is None:
            empty_steps = []
            for _ in sent2:
                empty_steps += ['']
            sent2 = empty_steps
            if step - old_step > 1:
                uttr = connect_sents([uttr, 'remove step %d to step %d.' % (old_step, step)], rule=1)   # remove the sql operation
            else:
                uttr = connect_sents([uttr, 'remove step %d.' % step], rule=1)
        elif correct_parse.operation == 'intersect':
            uttr = connect_sents([uttr, add(f"return the rows in {info('both of')} the results of step {old_step - 1} and the results of step {step - 1}.")], rule=2)
        elif correct_parse.operation == 'except':
            uttr = connect_sents([uttr, add(f"return the rows in the results of step {old_step - 1} {info('but not')} the results of step {step - 1}.")], rule=2)

        feedbacks = sent1 + sent2 + [uttr]
        step += 1

    elif not wrong_parse.exceptt.is_empty:
        wrong_sub = wrong_parse.exceptt.sub_query
        wrong_parse.exceptt.empty()
        correct_sub = None
        if correct_parse is None:
            correct_sub = None
        elif correct_parse.sub_query is not None:
            correct_sub = correct_parse.sub_query
            correct_parse.sub_query = None
            correct_parse.intersect.empty()
            correct_parse.union.empty()
            correct_parse.exceptt.empty()

        sent1, step = get_feedback(wrong_parse, correct_parse, step)
        old_step = step
        sent2, step = get_feedback(wrong_sub, correct_sub, step)
        uttr = ''
        if correct_sub is None:
            empty_steps = []
            for _ in sent2:
                empty_steps += ['']
            sent2 = empty_steps
            if step - old_step > 1:
                uttr = connect_sents([uttr, 'remove step %d to step %d.' % (old_step, step)], rule=1)   # remove the sql operation
            else:
                uttr = connect_sents([uttr, 'remove step %d.' % step], rule=1)
        elif correct_parse.operation == 'intersect':
            uttr = connect_sents([uttr, add(f"return the rows in {info('both of')} the results of step {old_step - 1} and the results of step {step - 1}.")], rule=2)
        elif correct_parse.operation == 'union':
            uttr = connect_sents([uttr, add(f"return the rows in {info('any of')} the results of step {old_step - 1} and the results of step {step - 1}.")], rule=2) # change the sql operator

        feedbacks = sent1 + sent2 + [uttr]
        step += 1
    
    elif not wrong_parse.limit.is_empty and wrong_parse.limit.val > 1:
        sent2 = ''
        if correct_parse is None:
            sent2 = '#undefined#'
        elif correct_parse.limit.is_empty:  # remove the limit clause
            uttr = 'all the results'
            sent2 = connect_sents([sent2, add(f"show me {uttr}.")], step, rule=4)
        elif correct_parse.limit.val == 1:  # change to largest / smallest
            pass    # This case is managed inside the main if
        elif wrong_parse.limit.val != correct_parse.limit.val:
            uttr = 'first %d' % correct_parse.limit.val
            sent2 = connect_sents([sent2, add(f"only show the {uttr} rows of the results.")], step, rule=5)
            correct_parse.limit.empty()
        elif wrong_parse.limit.val == correct_parse.limit.val:
            correct_parse.limit.empty() # No need to change
        # remove LIMIT from wrong_parse
        wrong_parse.limit.empty()
        sent1, step = get_feedback(wrong_parse, correct_parse, step)
        feedbacks = sent1 + [sent2]
        step += 1

    elif not wrong_parse.fromm.is_empty and isinstance(wrong_parse.fromm.tables[0], Query):
        # when nested sub query in FROM clause
        assert len(wrong_parse.fromm.tables) == 1, "Apart of the subquery, more things in the FROM clause"
        wrong_sub = wrong_parse.fromm.tables[0]
        sent3 = ''
        correct_tables = []
        if correct_parse is None:
            correct_sub = None
        elif correct_parse.fromm.is_empty:
            correct_sub = None
        elif isinstance(correct_parse.fromm.tables[0], Query):
            correct_sub = correct_parse.fromm.tables[0]
            correct_parse.fromm.empty()
        else:
            correct_sub = None
            correct_tables = correct_parse.fromm.tables
            correct_parse.fromm.empty()
        wrong_sub = correct_parse.fromm.table[0]
        wrong_parse.fromm.empty()
        very_old_step = step
        sent1, step = get_feedback(wrong_sub, correct_sub, step)    # explain the nested SQL first
        old_step = step
        sent2, step = get_feedback(wrong_parse, correct_parse, step, join_table=True)
        if correct_sub is None:     # remove the correction of nested SQL
            sent1 = ['' for _ in sent1]
            if old_step - very_old_step == 1:
                uttr = f"remove the step {very_old_step}."
            else:
                uttr = f"remove the step {very_old_step} to {old_step - 1}."
            sent3 = connect_sents([sent3, uttr], step, rule = 6)
            if len(correct_tables) > 0:
                uttr = add(f"use the {connect_words(correct_tables, 'table')}.")
                sent3 = connect_sents([sent3, uttr], step, rule=6)

        feedbacks = sent1 + sent2 + [sent3]
    
    elif not wrong_parse.fromm.is_empty and len(wrong_parse.fromm.tables) > 1: 
        # The JOIN ON step
        # Assume no nested SQL in FROM in this case
        assert not isinstance(wrong_parse.fromm.tables[0], Query), "subquery in non-single FROM clause"
        wrong_tables = set(wrong_parse.fromm.tables)
        correct_tables = set()
        sent1 = ''
        if not correct_parse is None:
            if not correct_parse.fromm.is_empty:
                if not isinstance(correct_parse.fromm.tables[0], Query):
                    correct_tables = set(correct_parse.fromm.tables)
                    correct_parse.fromm.empty()
                else:   # when the correct FROM table is SQL, add this nested SQL
                    sub_query = correct_parse.fromm.tables[0]
                    wrong_parse.fromm.empty()
                    correct_parse.fromm.empty()
                    explanation, sub_step = get_explanation(sub_query)
                    n = len(explanation)
                    if n == 1:
                        try:
                            uttr1 = simple_explain(sub_query)
                        except Exception:
                            if ABANDON:
                                config.ABANDONED[102] += 1
                            raise StructureError("have empty feedback because trying to modify a nested complex SQL query with more than one tables from FROM clause.")
                        uttr = add(f"use the information from the {uttr1}") + ' ' + sub(f"in place of the {connect_words(wrong_tables, 'table')}.")
                        explanation = []
                    else:
                        uttr = add(f"use the information of the results of extra step {sub_step - 1}") + ' ' + sub(f"in place of the {connect_words(wrong_tables, 'table')}.")
                        explanation = add_step(explanation, step="extra step")
                    sent1 = ''
                    sent1 = connect_sents([sent1, uttr], step, rule=7)  # add a SQL in FROM
                    if not RULES[7]:
                        explanation = []

                    sent2, step = get_feedback(wrong_parse, correct_parse, step + 1, join_table=True)
                    
                    return explanation + [sent1] + sent2, step
        # remove from wrong_parse
        wrong_parse.fromm.empty()

        added_tables = list(correct_tables - wrong_tables)
        removed_tables = list(wrong_tables - correct_tables)

        if len(added_tables) + len(removed_tables) == 0:
            sent1 = ''
        else:            
            if len(added_tables) > 0 and len(removed_tables) > 0:
                add_uttr = add('use %s' % connect_words(added_tables, 'table')) + ' ' + sub(f"in place of {connect_words(removed_tables, 'table')}")
                sent1 = connect_sents([sent1, add_uttr], step, rule=8)      # change tables
            elif len(removed_tables) > 0:
                remove_uttr = add('do not use the %s.' % connect_words(removed_tables, 'table'))
                sent1 = connect_sents([sent1, sub(remove_uttr)], step, rule=9)   # remove tables
            elif len(added_tables) > 0:
                uttr = f"additionally use the information from the {connect_words(added_tables, 'table')}."
                uttr = add(uttr)
                sent1 = connect_sents([sent1, uttr], step, rule=10)         # add tables
            else:
                raise Exception('new pattern found!')

        sent2, step = get_feedback(wrong_parse, correct_parse, step + 1, join_table=True)
        feedbacks = [sent1] + sent2
    
    else:
        bool_groupby = not wrong_parse.group_by.is_empty
        bool_orderby = not wrong_parse.order_by.is_empty
        bool_having = not wrong_parse.having.is_empty
        bool_where = not wrong_parse.where.is_empty
                                
        table_correction = ''  # table correction is described in the JOIN step
        table_feedbacks = []    # for the nested SQL in FROM clause
        if not join_table:
            wrong_tables = set(wrong_parse.fromm.tables)
            correct_tables = set()
            if not correct_parse is None:
                if not correct_parse.fromm.is_empty:
                    uttr = ''
                    if isinstance(correct_parse.fromm.tables[0], Query):
                        assert len(correct_parse.fromm.tables) == 1, "more than one table in FROM clause while FROM is nested!"
                        explanation, sub_step = get_explanation(correct_parse.fromm.tables[0])
                        table_feedbacks = add_step(explanation, 'extra step')
                        uttr = connect_sents([uttr, f"additionally use the information from the results of extra step {sub_step - 1}."], rule=3)
                        uttr = add(uttr)
                        n = len(explanation)
                        if n == 1:
                            explanation = simple_explain(correct_parse.fromm.tables[0])
                            uttr = connect_sents([uttr, f"additionally use the information of the {explanation}."], rule=3)
                            uttr = add(uttr)
                            table_feedbacks = []

                        correct_tables = set([uttr])
                    else:
                        correct_tables = set(correct_parse.fromm.tables)
                    correct_parse.fromm.empty()
            added_tables = list(correct_tables - wrong_tables)
            removed_tables = list(wrong_tables - correct_tables)
            if len(added_tables) + len(removed_tables) > 0:
                if len(added_tables) > 0 and len(removed_tables) > 0:
                    table_correction = add('use %s' % connect_words(added_tables, 'table')) + ' ' + sub('in place of %s.' % connect_words(removed_tables, 'table'))
                elif len(added_tables) > 0:
                    if len(list(wrong_tables)) == 1:
                        table_correction = add('additionally use the information from the %s' % connect_words(added_tables, 'table')) + ' ' + sub('besides the %s.' % connect_words(wrong_tables, 'table'))
                    else:
                        table_correction = 'additionally use the information from the %s.' % connect_words(added_tables, 'table')
                        table_correction = add(table_correction)
                else:
                    table_correction = 'do not use the %s.' % connect_words(removed_tables, 'table')
                    table_correction = add(table_correction)
        
        # find out the different ColUnit, ValUnit or SelectArgu in the SELECT clause
        wrong_cols = set(wrong_parse.select.args)
        correct_cols = set()
        wrong_distinct = wrong_parse.select.distinct
        correct_distinct = None
        if not correct_parse is None:
            if not correct_parse.select.is_empty:
                correct_cols = set(correct_parse.select.args)
                correct_distinct = correct_parse.select.distinct
                correct_parse.select.empty()
        added_cols = list(correct_cols - wrong_cols)
        removed_cols = list(wrong_cols - correct_cols)

        # Change agg operaion of a ColUnit
        changed = []
        changed_cols = []
        for added in added_cols:
            for rm in removed_cols:
                if isinstance(added, ColUnit) and isinstance(rm, ColUnit):
                    if added.name == rm.name:   # change agg
                        uttr = add(f"find {connect_words([added])}") + ' ' + sub(f"in place of {connect_words([rm], mode='sub')}.")
                        changed += [added, rm]
                        changed_cols += [uttr]

        added_cols = list(set(added_cols) - set(changed))
        removed_cols = list(set(removed_cols) - set(changed))
        
        # select ...
        sel_correction = ''
        if len(added_cols) + len(removed_cols) + len(changed_cols) > 0:
            if len(added_cols) > 0 and len(removed_cols) > 0:
                sel_correction = add(f"find {connect_words(added_cols, mode='add')}") + ' ' + sub(f"in place of {connect_words(removed_cols, mode='sub')}.")
            elif len(removed_cols) == 0 and len(added_cols) > 0:
                sel_correction = 'additionally find %s.' % connect_words(added_cols)
                sel_correction = add(sel_correction)
            elif len(added_cols) == 0 and len(removed_cols) > 0:
                sel_correction = 'do not return %s.' % connect_words(removed_cols)
                sel_correction = add(sel_correction)
        sel_correction = connect_sents([sel_correction] + changed_cols)
                
        # Change the distinct in SELECT
        if wrong_distinct != correct_distinct:
            if correct_distinct == True:
                if len(sel_correction) == 0:
                    sel_correction = add('make sure no repetition in the results.')
                else:
                    sel_correction += f" {add('make sure no repetition in the results.')}"
            elif correct_distinct == False or (correct_distinct is None and wrong_distinct == True):
                if len(sel_correction) == 0:
                    sel_correction = f"{add('permit repetitions in the results.')}"
                else:
                    sel_correction += f" {add('permit repetitions in the results.')}"
        
        ######################################################################################## WHERE Correction #############################################
        # the WHERE clause correction, there might be nested quiries within it
        where_feedbacks = table_feedbacks  # for the nested SQL in WHERE clause, including the nested SQL in FROM clause
        where_correction = ''
        where_conds_wrong = []
        wrong_where_is_nested = False
        where_or_pairs_wrong = set()
        if bool_where:
            where_or_pairs_wrong = wrong_parse.where.or_pairs
            if not correct_parse is None and not correct_parse.where.is_empty:
                wrong_connectors = wrong_parse.where.connectors
                correct_connectors = correct_parse.where.connectors
                if 'and' not in wrong_connectors and 'and' in correct_connectors and len(wrong_connectors) > 0:
                    where_correction = connect_sents([where_correction, f"{add('you should consider both of the conditions rather than either of them.')}"])
                elif 'or' not in wrong_connectors and 'or' in correct_connectors and len(wrong_connectors) > 0:
                    where_correction = connect_sents([where_correction, f"{add('you shoud consider either of the conditions rather than both of them.')}"])
            for cond in wrong_parse.where.conds:
                if cond.is_nested:
                    wrong_where_is_nested = True
                    # There are only 7 cases in gold quiries from 7480 training data which have more than one nested condition in WHERE 
                    if len(cond.sub_quiries) > 1: 
                        if ABANDON:
                            config.ABANDONED[103] += 1
                        raise StructureError("more than one subquires in one cond_unit")
                    if isinstance(cond.val0, Query) or not isinstance(cond.val1, Query): raise Exception("new format of cond_unit founded!")
                    # Assume only one nested SQL in WHERE CLAUSE
                    correct_sub = None
                    wrong_sub = cond.sub_quiries[0]
                    if correct_parse is not None:   # Find the nested SQL in the correct parse
                        if not correct_parse.where.is_empty:
                            for correct_cond in correct_parse.where.conds:
                                if correct_cond.is_nested:
                                    correct_sub = correct_cond.sub_quiries[0]
                                    break
                    old_step = step
                    where_feedbacks, step = get_feedback(wrong_sub, correct_sub, step)
                    if correct_sub is None:
                        # Remove the SQL condition
                        if cond.op == 'between':
                            if cond.negation:
                                where_correction = connect_sents([where_correction, add(f"remove the {cond.val0} is not between the results of step {step - 1} and {cond.val2} condition.")], rule=21) 
                            else:
                                where_correction = connect_sents([where_correction, add(f"remove the {cond.val0} is between the results of step {step - 1} and {cond.val2} condition.")], rule=21) 
                        else:
                            if cond.negation:
                                where_correction = connect_sents([where_correction, add(f"remove the {cond.val0} is not {cond.op_uttr()} the results of step {step - 1} condition.")], rule=21)
                            else:
                                where_correction = connect_sents([where_correction, add(f"remove the {cond.val0} is {cond.op_uttr()} the results of step {step - 1} condition.")], rule=21)
                        where_feedbacks = ['' for _ in where_feedbacks]

                    else:   # when SQL is in both wrong and correct WHERE is nested
                        if cond.val0 != correct_cond.val0 or cond.op != correct_cond.op or cond.negation != correct_cond.negation or cond.val2 != correct_cond.val2:
                            # the nested condition is wrong, change the SQL condition
                            if correct_cond.op == 'between':
                                if correct_cond.negation:
                                    where_correction = connect_sents([where_correction, add(f'consider the {correct_cond.val0} is not between the results of step {str(step - 1)} and {correct_cond.val2}')]) 
                                else:
                                    where_correction = connect_sents([where_correction, add(f'make sure the {correct_cond.val0} is between the results of step {str(step - 1)} and {correct_cond.val2}')]) 
                            else:
                                if correct_cond.negation:
                                    where_correction = connect_sents([where_correction, add(f'make sure the {correct_cond.val0} is not {correct_cond.op_uttr()} the results of step {str(step - 1)}')])
                                else:
                                    where_correction = connect_sents([where_correction, add(f'make sure the {correct_cond.val0} is {correct_cond.op_uttr()} the results of step {str(step - 1)}')]) 
                            if cond.op == 'between':
                                if cond.negation:
                                    where_correction = connect_sents([where_correction, sub(f"the {cond.val0} is not between the results of step {step - 1} and {cond.val2}.")]) 
                                else:
                                    where_correction = connect_sents([where_correction, sub(f"the {cond.val0} is between the results of step {step - 1} and {cond.val2}.")]) 
                            else:
                                if cond.negation:
                                    where_correction = connect_sents([where_correction, sub(f"the {cond.val0} is not {cond.op_uttr()} the results of step {step - 1}.")]) 
                                else:
                                    where_correction = connect_sents([where_correction, sub(f"the {cond.val0} is {cond.op_uttr()} the results of step {step - 1}.")]) 

                else:       # when the wrong cond is not nested
                    where_conds_wrong +=[cond]
        
        where_conds_correct = []
        where_or_pairs_correct = set()
        if correct_parse is not None:
            if not correct_parse.where.is_empty:
                where_or_pairs_correct = correct_parse.where.or_pairs
                for cond in correct_parse.where.conds:
                    if not cond.is_nested:
                        where_conds_correct += [cond]
                    elif not wrong_where_is_nested:   # add one nested SQL in WHERE
                        if isinstance(cond.val0, Query) or not isinstance(cond.val1, Query): raise Exception("new format of cond_unit founded!")
                        correct_sub = cond.val1
                        explanation, sub_step = get_explanation(correct_sub)
                        if len(explanation) > 1:    # when the correct sub_sql is more than one step
                            explanation = add_step(explanation, 'extra step')
                            where_feedbacks += explanation
                            uttr1 = ''
                            if cond.op == 'between':
                                if cond.negation:
                                    uttr = f"{cond.val0} is not between the results of step {step - 1} and {cond.val2}"
                                else:
                                    uttr = f"{cond.val0} is between the results of step {step - 1} and {cond.val2}"
                                uttr = add(uttr)
                            else:
                                if cond.negation:
                                    uttr = f"{cond.val0} is not {cond.op_uttr()} the results of step {step - 1}"
                                else:
                                    uttr = f"{cond.val0} is {cond.op_uttr()} the results of step {step - 1}"
                                uttr = add(uttr)
                            uttr1 = connect_sents([uttr1, uttr], rule=20)   # rule 20: add complex sub-query in WHERE condition
                            where_conds_correct += [uttr1]
                            for cond2 in correct_parse.where.conds:
                                if hash(cond2) + hash(cond) in where_or_pairs_correct:
                                    where_or_pairs_correct.add(hash(uttr) + hash(cond2))
                        else:                       # when the correct sub_sql is 1 step explanation
                            explanation = simple_explain(correct_sub)
                            if cond.op == 'between':
                                if cond.negation:
                                    uttr = f"{cond.val0} is not between the {explanation} and {cond.val2}"
                                else:
                                    uttr = f"{cond.val0} is between the {explanation} and {cond.val2}"
                            else:
                                if cond.negation:
                                    uttr = f"{cond.val0} is not {cond.op_uttr()} {explanation}"
                                else:
                                    uttr = f"{cond.val0} is {cond.op_uttr()} {explanation}"
                            uttr = add(uttr)
                            where_conds_correct += [uttr]

                correct_parse.where.empty()

        added_conds = list(set(where_conds_correct) - set(where_conds_wrong))
        removed_conds = list(set(where_conds_wrong) - set(where_conds_correct))

        def connect_where_conditions(conds: list, or_pairs: set, mode: str='') -> str:
            # This function is used to connect conditions with correct "AND" or "OR" operators.
            if not config.SHOW_TAG:
                mode = ''
            original_conds = set(conds)
            conds_pairs = []
            for i in range(len(conds)):
                for j in range(i + 1, len(conds)):
                    if hash(conds[i]) + hash(conds[j]) in or_pairs:
                        conds_pairs += [(conds[i], conds[j])]
                        original_conds -= set([conds[i]])
                        original_conds -= set([conds[j]])

            uttr = ''
            original_conds = list(original_conds)
            if len(original_conds) == 0: uttr = ''
            elif len(original_conds) == 1: 
                if mode == 'add':
                    uttr = ADD_TAG1 + str(original_conds[0]) + ADD_TAG2
                elif mode == 'sub':
                    uttr = SUB_TAG1 + str(original_conds[0]) + SUB_TAG2
                else:
                    uttr = str(original_conds[0])
            else:
                for cond in original_conds[:-1]:
                    if mode == 'add':
                        uttr += f"{ADD_TAG1}{cond}{ADD_TAG2} and "
                    elif mode == 'sub':
                        uttr += f"{SUB_TAG1}{cond}{SUB_TAG2} and "
                    else:
                        uttr += f"{cond} and "
                if mode == 'add':
                    uttr += f"{ADD_TAG1}{original_conds[-1]}{ADD_TAG2}"
                elif mode == 'sub':
                    uttr += f"{SUB_TAG1}{original_conds[-1]}{SUB_TAG2}"
                else:
                    uttr += f"{original_conds[-1]}"
            
            for pair in conds_pairs:
                if uttr == '':
                    if mode == 'add':
                        uttr = f"{ADD_TAG1}{pair[0]}{ADD_TAG2} or {ADD_TAG1}{pair[1]}{ADD_TAG2}"
                    elif mode == 'sub':
                        uttr = f"{SUB_TAG1}{pair[0]}{SUB_TAG2} or {SUB_TAG1}{pair[1]}{SUB_TAG2}"
                    else:
                        uttr = f"{pair[0]} or {pair[1]}"
                else:
                    if mode == 'add':
                        uttr += f" and {ADD_TAG1}{pair[0]}{ADD_TAG2} or {ADD_TAG1}{pair[1]}{ADD_TAG2}"
                    elif mode == 'sub':
                        uttr += f" and {SUB_TAG1}{pair[0]}{SUB_TAG2} or {SUB_TAG1}{pair[1]}{SUB_TAG2}"
                    else:
                        uttr += f" and {pair[0]} or {pair[1]}"
            
            return uttr
    
        if len(added_conds) > 0 and len(removed_conds) > 0:
            uttr = add(f"consider the {connect_where_conditions(added_conds, or_pairs=where_or_pairs_correct)} conditions") + ' ' + sub(f"in place of the {connect_where_conditions(removed_conds, or_pairs=where_or_pairs_wrong)} conditions.")
            where_correction = connect_sents([where_correction, uttr])
        elif len(added_conds) > 0:
            where_correction = connect_sents([where_correction, add('additionally make sure that %s.' % connect_where_conditions(added_conds, or_pairs=where_or_pairs_correct))]) 
        elif len(removed_conds) > 0:
            where_correction = connect_sents([where_correction, add('remove the %s conditions.' % connect_where_conditions(removed_conds, or_pairs=where_or_pairs_correct))]) 
        ######################################################################################## WHERE Correction Finished ####################################


        # Find added and removed GROUPBY columns
        wrong_groupby_cols = set()
        if bool_groupby:
            wrong_groupby_cols = set(wrong_parse.group_by.args)
        correct_groupby_cols = set()
        if correct_parse is not None:
            if not correct_parse.group_by.is_empty:
                correct_groupby_cols = set(correct_parse.group_by.args)
                correct_parse.group_by.empty()
        grouby_added = list(correct_groupby_cols - wrong_groupby_cols)
        grouby_removed = list(wrong_groupby_cols - correct_groupby_cols)        

        # Find the added and removed ORDERBY columns, also for LIMIT clause
        added_orderby_cols = []
        removed_orderby_cols = []
        wrong_orderby_cols = set()
        correct_orderby_cols = set()
        wrong_orderby_direction = None
        wrong_orderby_est = None
        wrong_limit_value = None
        if bool_orderby:
            wrong_orderby_cols = set(wrong_parse.order_by.args)
            wrong_orderby_direction = {"asc": "ascending", "desc": "descending"}[wrong_parse.order_by.dir]
            wrong_orderby_est = {"asc": "smallest", "desc": "largest"}[wrong_parse.order_by.dir]
            wrong_limit_value = wrong_parse.limit.val
            if wrong_parse.limit.is_empty:
                wrong_limit_value = None
            if not (wrong_limit_value is None or wrong_limit_value == 1): raise Exception("the LIMIT larger than 1 should processed outside")  # the LIMIT larger than 1 is processed outside

        correct_orderby_direction = None
        correct_orderby_est = None
        correct_limit_value = None
        if correct_parse is not None:
            if not correct_parse.order_by.is_empty:
                correct_orderby_cols = set(correct_parse.order_by.args)
                correct_orderby_direction = {"asc": "ascending", "desc": "descending"}[correct_parse.order_by.dir]
                correct_orderby_est = {"asc": "smallest", "desc": "largest"}[correct_parse.order_by.dir]
                correct_parse.order_by.empty()
                if not correct_parse.limit.is_empty:
                    correct_limit_value = correct_parse.limit.val
                    correct_parse.limit.empty()

        added_orderby_cols = list(correct_orderby_cols - wrong_orderby_cols)
        removed_orderby_cols = list(wrong_orderby_cols - correct_orderby_cols)

        # orderby Correction
        orderby_correction = ''
        if bool_orderby and len(list(correct_orderby_cols)) == 0:   # remove the orderby clause
            if wrong_limit_value == 1:
                uttr = add(f"you should not find the {wrong_orderby_est} of the results.")
            else:
                uttr = add('you should not order the results.')
            wrong_parse.order_by.empty()
        elif not bool_orderby and len(list(correct_orderby_cols)) > 0: # adding the orderby clause
            if correct_limit_value == 1: 
                uttr = f"find the result with the {correct_orderby_est} {connect_words(added_orderby_cols)}."          # adding the order case
                uttr = add(uttr)
            elif correct_limit_value is None: 
                uttr = f"order the results {correct_orderby_direction} by {connect_words(added_orderby_cols)}."   # adding the largest/smallest case
                uttr = add(uttr)
            elif correct_limit_value > 1: 
                uttr = f"order the results {correct_orderby_direction} by {connect_words(added_orderby_cols)}."
                uttr = add(uttr)
            else: uttr = "#undefined#"
        elif wrong_limit_value is None and correct_limit_value == 1:    # add or change to the LIMIT 1 clause
            uttr = f"find the result with the {correct_orderby_est} {connect_words(correct_orderby_cols)}."
            uttr = add(uttr)
        elif len(added_orderby_cols + removed_orderby_cols) > 0 and wrong_orderby_direction != correct_orderby_direction:
            uttr = add(f"order the results {correct_orderby_direction} by {connect_words(correct_orderby_cols)}") + ' ' + sub(f"in place of ordering {wrong_orderby_direction} by {connect_words(wrong_orderby_cols)}.")
        elif len(added_orderby_cols) > 0 and len(removed_orderby_cols) > 0:
            uttr = add(f"order the results by {connect_words(added_orderby_cols)}") + ' ' + sub(f"in place of {connect_words(removed_orderby_cols)}.")
        elif len(added_orderby_cols) > 0:
            uttr = f"additionally order the results by {connect_words(added_orderby_cols)}."
            uttr = add(uttr)
        elif len(removed_orderby_cols) > 0:
            uttr = f"you should not order the results by {connect_words(removed_orderby_cols)}."
            uttr = sub(uttr)
        elif wrong_orderby_direction != correct_orderby_direction:
            if correct_limit_value == 1 and wrong_limit_value == 1:
                uttr = add(f"use the {correct_orderby_est}") + ' ' + sub(f"in place of {wrong_orderby_est}.")
            else:
                uttr = f"order the results {correct_orderby_direction}."
                uttr = add(uttr)
        else:
            uttr = ''

        if correct_limit_value is not None and correct_limit_value > 1: # adding or change LIMIT 1 to the LIMIT > 1 case
            uttr1 = f"first {correct_limit_value}"
            uttr = connect_sents([uttr, add(f"only show me the {uttr1} results.")])
            uttr = add(uttr)
        if wrong_limit_value == 1 and correct_limit_value is None:  # remove the LIMIT 1 case
            uttr = connect_sents([uttr, f"show me {'all the results'}."])
            uttr = add(uttr)
        orderby_correction = connect_sents([orderby_correction, uttr])
        wrong_parse.order_by.empty()
        wrong_parse.limit.empty()
        if correct_parse is not None:
            correct_parse.order_by.empty()
            correct_parse.limit.empty()

        # Correct the having clause
        wrong_having_col = ''
        wrong_having_description = ''
        if bool_having:
            if not len(wrong_parse.having.conds) == 1: 
                if ABANDON:
                    config.ABANDONED[104] += 1
                raise StructureError("assume only one cond_unit in HAVING")
            if wrong_parse.having.conds[0].is_nested: 
                if ABANDON:
                    config.ABANDONED[105] += 1
                raise StructureError("assume no nested SQL in the HAVING clause")
            wrong_having_col = str(wrong_parse.having.conds[0].val0)
            wrong_having_description = wrong_parse.having.conds[0].description
            wrong_parse.having.empty()

        correct_havingc_col = ''
        correct_having_description = ''
        if not correct_parse is None:
            if not correct_parse.having.is_empty:
                if not len(correct_parse.having.conds) == 1: 
                    if ABANDON:
                        config.ABANDONED[104] += 1
                    raise StructureError("assume only one cond_unit in HAVING")
                if correct_parse.having.conds[0].is_nested: 
                    if ABANDON:
                        config.ABANDONED[105] += 1
                    raise StructureError("assume no nested SQL in the HAVING clause")
                correct_havingc_col = str(correct_parse.having.conds[0].val0)
                correct_having_description = correct_parse.having.conds[0].description
                correct_parse.having.empty()
                    

        # This is for the case there are no GROUPBY and no HAVING in wrong clause
        # select ... from ... [where] ... [order by]
        if not bool_having and not bool_groupby:
            sent1 = connect_sents(['', table_correction, where_correction], step)
            if len(grouby_added) > 0:   # adding an entire groupby clause
                    if len(correct_havingc_col) > 0:
                        uttr = f"find for each value of {connect_words(correct_groupby_cols)} whose {correct_havingc_col} {correct_having_description}."
                        uttr = add(uttr)
                    else:
                        uttr = f"find for each value of {connect_words(correct_groupby_cols)}."
                        uttr = add(uttr)
                    # if step > 1:
                    #     uttr = f"right before step {step}, {uttr}"
            else:
                uttr = ''
            sent1 = connect_sents([sent1, uttr, sel_correction, orderby_correction], step)
                        
            return where_feedbacks + [sent1], step + 1


        # when there are both WHERE and GROUPBY, we use an extra step to explain / correct the WHERE / FROM
        if bool_groupby and bool_where:
            sent = connect_sents(['', table_correction, where_correction], step)
            where_feedbacks += [sent]
            table_correction = ''   # The tables have already corrected
            where_correction = ''   # The WHERE has already corrected
            step += 1        

        # select ... [where ...] group by ...
        # e.g., SELECT Employee_ID , Count ( * ) FROM Employees GROUP BY Employee_ID ->
        # find each value of Employee_ID in Employees table along with the number of the corresponding rows to each value
        if bool_groupby and not (bool_orderby or bool_having):
            if len(correct_havingc_col) > 0 and len(list(correct_groupby_cols)) > 0: 
                if ABANDON:
                    config.ABANDONED[106] += 1
                raise StructureError('Adding HAVING cases are ignored!')
            sent = ''
            
            if len(list(correct_groupby_cols)) == 0:    # remove the entire GROUP BY clause
                uttr = f"do not find for each value of {connect_words(wrong_groupby_cols)}."
                uttr = add(uttr)
            elif correct_parse.group_by != wrong_parse.group_by:
                if len(grouby_added) > 0 and len(grouby_removed) > 0:
                    uttr = add(f"find for each value of {connect_words(grouby_added)}") + ' ' + sub(f"in place of {connect_words(grouby_removed, mode='sub')}.")
                elif len(grouby_added) > 0 and len(grouby_removed) == 0:
                    uttr = add(f"additionally find for each value of {connect_words(grouby_added)}.")
                else:
                    uttr = add(f"find for each value of {connect_words(correct_groupby_cols)}.")
            else:
                uttr = ''

            sent = connect_sents([sent, table_correction, where_correction, uttr, sel_correction, orderby_correction], step)

            return where_feedbacks + [sent], step + 1
        
        # select ... [where ...] group by ... order by ...
        if bool_groupby and bool_orderby and not bool_having:
            if len(correct_havingc_col) > 0 and len(list(correct_groupby_cols)) > 0: 
                if ABANDON:
                    config.ABANDONED[106] += 1
                raise StructureError('Adding HAVING cases are ignored!')
            sent1 = ''  # GROUP BY, FROM
            sent2 = ''  # SELECT, ORDER BY

            if len(list(correct_groupby_cols)) == 0:    # remove the entire GROUP BY clause
                uttr = f"do not find for each value of {connect_words(wrong_groupby_cols)}."
                uttr = add(uttr)
            elif correct_parse.group_by != wrong_parse.group_by:
                if len(grouby_added) > 0 and len(grouby_removed) > 0:
                    uttr = add(f"find for each value of {connect_words(grouby_added)}") + ' ' + sub(f"in place of {connect_words(grouby_removed, mode='sub')}.")
                elif len(grouby_added) > 0 and len(grouby_removed) == 0:
                    uttr = add(f"additionally find for each value of {connect_words(grouby_added)}.")
                else:
                    uttr = add(f"find for each value of {connect_words(correct_groupby_cols)}.")
            else:
                uttr = ''

            sent1 = connect_sents([sent1, table_correction, where_correction, uttr], step)
            sent2 = connect_sents([sent2, sel_correction, orderby_correction], step+1)

            return where_feedbacks + [sent1, sent2], step + 2
        
        # select ... [where ...] group by ... having ...
        if bool_groupby and bool_having and not bool_orderby:
            if len(correct_havingc_col) == 0 and len(list(correct_groupby_cols)) > 0: 
                if ABANDON:
                    config.ABANDONED[107] += 1
                raise StructureError('Removing HAVING cases are ignored!')
            sent1 = ''  # GROUP BY, HAVING.col
            sent2 = ''  # SELECT, FROM, HAVING.des

            if len(list(correct_groupby_cols)) == 0:    # remove the entire GROUP BY clause
                uttr1 = add(f"do not find for each value of {connect_words(wrong_groupby_cols)}.")
                uttr2 = ''
            elif correct_parse.group_by != wrong_parse.group_by and correct_parse.having == wrong_parse.having:    # GROUP BY is wong and HAVIGN is correct
                if len(grouby_removed) > 0 and len(grouby_added) > 0:
                    uttr1 = add(f"find for value of {connect_words(grouby_added)}") + ' ' + sub(f"in place of {connect_words(grouby_removed)}.")
                    uttr2 = ''
                elif len(grouby_added) > 0 and len(grouby_removed) == 0:
                    uttr1 = add(f"additionally find for each value of {connect_words(grouby_added)}.")
                    uttr2 = ''
                else:
                    uttr1 = add(f"find for each value of {connect_words(correct_groupby_cols)}.")
                    uttr2 = ''
            elif correct_parse.group_by != wrong_parse.group_by and correct_parse.having != wrong_parse.having:     # GROUP BY and HAVING are wrong
                uttr1 = add(f"find for each value of {connect_words(correct_groupby_cols)} whose {correct_havingc_col} {correct_having_description}.")
                uttr2 = ''
            elif correct_parse.having != wrong_parse.having:    # GROUP BY is correct and HAVING is wrong
                if correct_parse.having.conds[0].val0 != wrong_parse.having.conds[0].val0:
                    uttr1 = add(f"find the {correct_havingc_col} for each value of {connect_words(correct_groupby_cols)}.")
                else:
                    uttr1 = ''
                if correct_parse.having.conds[0].val1 != wrong_parse.having.conds[0].val1 or correct_parse.having.conds[0].val2 != wrong_parse.having.conds[0].val2 or correct_parse.having.conds[0].val2 != wrong_parse.having.conds[0].val2 or correct_parse.having.conds[0].op != wrong_parse.having.conds[0].op:
                    uttr2 = add(f"make sure that the corresponding value in step {step} {correct_having_description}.")
                else:
                    uttr2 = ''
            else:
                uttr1 = ''
                uttr2 = ''
            
            sent1 = connect_sents([sent1, table_correction, uttr1], step)
            sent2 = connect_sents([sent2, where_correction, uttr2, sel_correction, orderby_correction], step+1)

            return where_feedbacks + [sent1, sent2], step + 2

        # Assume the HAVING and ORDER BY will not appear together.
    
        raise Exception("New patterns discovered!")
        
    return feedbacks, step

