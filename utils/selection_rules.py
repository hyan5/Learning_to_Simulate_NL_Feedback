ruleWords = ['%', 'maximum', 'minimum', 'number', 'sum', 'average', 'minus', 'times', 'divided', 'by', 'plus', 'all', 'distinct', 'not', 'between', 'and', 'or', 'greater', 'less', 'than', 'to', 'equals', 'one', 'of', 'like', 'desc', 'asc', 'remove', 'add', 'descending', 'ascending']
KEY_PHRASE_MAP = {
    str(['stuid']): [['stuid'], ['student', 'id'], ['stu', 'id']],
    str(['distinct']): [["distinct"], ["distinctive"], ["distinctively"], ["repeat"], ["repeated"], ["repetition"], ["repetitious"], ["duplicate"]],
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


# <select> add a new column -> Column name + (optional) table name
def rule1(edits: dict) -> list:
    addArgs = edits['select']['adds']
    rmArgs = edits['select']['rms']

    adds = []
    rms = []

    for arg in addArgs:
        phrases = arg.lower().strip().split()
        for phrase in phrases:
            if not phrase in ruleWords:
                adds += [phrase]

    for arg in rmArgs:
        phrases = arg.lower().strip().split()
        for phrase in phrases:
            if not phrase in ruleWords:
                rms += [phrase]

    adds = set(adds)
    rms = set(rms)

    phrases = list(adds - rms)

    keyPhrases = []
    for phrase in phrases:
        phrase = phrase.replace('_', ' ').replace('-', ' ').replace('"', ' ').replace("'", ' ').replace(',', ' ').lower().split()
        keyPhrases += [phrase]

    return keyPhrases

# <select> remove a column with no new added columns -> Wrong column name
def rule2(edits: dict) -> list:
    addArgs = edits['select']['adds']
    rmArgs = edits['select']['rms']

    adds = []
    rms = []

    for arg in addArgs:
        phrases = arg.lower().strip().split()
        for phrase in phrases:
            if not phrase in ruleWords:
                adds += [phrase]
    
    for arg in rmArgs:
        phrases = arg.lower().strip().split()
        for phrase in phrases:
            if not phrase in ruleWords:
                rms += [phrase]

    adds = set(adds)
    rms = set(rms)

    news = list(adds - rms)
    if len(news) > 0:
        return []

    phrases = list(rms - adds)

    keyPhrases = []
    for phrase in phrases:
        phrase = phrase.replace('_', ' ').replace('-', ' ').replace('"', ' ').replace("'", ' ').replace(',', ' ').lower().split()
        keyPhrases += [phrase]

    return keyPhrases

# <select> add distinct 
def rule3(edits: dict) -> list:
    addArgs = edits['select']['adds']


    if len(addArgs) == 0:
        return []
    phrase = addArgs[0].lower().strip()
    if phrase == 'distinct':
        return ['distinct']

    return []

# <select> remove distinct
def rule4(edits: dict) -> list:
    addArgs = edits['select']['rms']


    if len(addArgs) == 0:
        return []
    phrase = addArgs[0].lower().strip()
    if phrase == 'distinct':
        return ['distinct']

    return []

# <where> add / revise a column name or value
def rule5(edits: dict) -> list:
    addArgs = edits['where']['adds']
    rmArgs = edits['where']['rms']

    adds = []
    rms = []

    for arg in addArgs:
        phrases = arg.lower().strip().split()
        for phrase in phrases:
            if not phrase in ruleWords:
                adds += [phrase]

    for arg in rmArgs:
        phrases = arg.lower().strip().split()
        for phrase in phrases:
            if not phrase in ruleWords:
                rms += [phrase]

    adds = set(adds)
    rms = set(rms)

    phrases = list(adds - rms)

    keyPhrases = []
    for phrase in phrases:
        phrase = phrase.replace('_', ' ').replace('-', ' ').replace('"', ' ').replace("'", ' ').replace(',', ' ').replace('%', ' ').lower().split()
        if not 'subs1' in phrase and not 'subs2' in phrase:
            keyPhrases += [phrase]

    return keyPhrases

# <groupby> add / replace a column -> the added column name
def rule7(edits:dict) -> list:
    addArgs = edits['groupBy']['adds']
    rmArgs = edits['groupBy']['rms']

    adds = []
    rms = []

    for arg in addArgs:
        phrases = arg.lower().strip().split()
        for phrase in phrases:
            if not phrase in ruleWords:
                adds += [phrase]

    for arg in rmArgs:
        phrases = arg.lower().strip().split()
        for phrase in phrases:
            if not phrase in ruleWords:
                rms += [phrase]

    adds = set(adds)
    rms = set(rms)

    phrases = list(adds - rms)

    keyPhrases = []
    for phrase in phrases:
        phrase = phrase.replace('_', ' ').replace('-', ' ').replace('"', ' ').replace("'", ' ').replace(',', ' ').lower().split()
        keyPhrases += [phrase]

    return keyPhrases

# <groupBy> remove a column -> removed column name
def rule8(edits:dict) -> list:
    addArgs = edits['groupBy']['adds']
    rmArgs = edits['groupBy']['rms']

    adds = []
    rms = []

    for arg in addArgs:
        phrases = arg.lower().strip().split()
        for phrase in phrases:
            if not phrase in ruleWords:
                adds += [phrase]

    for arg in rmArgs:
        phrases = arg.lower().strip().split()
        for phrase in phrases:
            if not phrase in ruleWords:
                rms += [phrase]

    adds = set(adds)
    rms = set(rms)

    phrases = list(rms - adds)

    keyPhrases = []
    for phrase in phrases:
        phrase = phrase.replace('_', ' ').replace('-', ' ').replace('"', ' ').replace("'", ' ').replace(',', ' ').lower().split()
        keyPhrases += [phrase]

    return keyPhrases

# <having> add / revise a condition's column name or value
def rule9(edits:dict) -> list:
    addArgs = edits['having']['adds']
    rmArgs = edits['having']['rms']

    adds = []
    rms = []

    for arg in addArgs:
        phrases = arg.lower().strip().split()
        for phrase in phrases:
            if not phrase in ruleWords:
                adds += [phrase]

    for arg in rmArgs:
        phrases = arg.lower().strip().split()
        for phrase in phrases:
            if not phrase in ruleWords:
                rms += [phrase]

    adds = set(adds)
    rms = set(rms)

    phrases = list(adds - rms)

    keyPhrases = []
    for phrase in phrases:
        phrase = phrase.replace('_', ' ').replace('-', ' ').replace('"', ' ').replace("'", ' ').replace(',', ' ').replace('%', ' ').lower().split()
        if not 'subs1' in phrase and not 'subs2' in phrase:
            keyPhrases += [phrase]

    return keyPhrases

# <orderBy> add / revise a column -> target column name
def rule11(edits: dict) -> list:
    addArgs = edits['orderBy']['adds']
    rmArgs = edits['orderBy']['rms']

    adds = []
    rms = []

    for arg in addArgs:
        phrases = arg.lower().strip().split()
        for phrase in phrases:
            if not phrase in ruleWords:
                adds += [phrase]

    for arg in rmArgs:
        phrases = arg.lower().strip().split()
        for phrase in phrases:
            if not phrase in ruleWords:
                rms += [phrase]

    adds = set(adds)
    rms = set(rms)

    phrases = list(adds - rms)

    keyPhrases = []
    for phrase in phrases:
        phrase = phrase.replace('_', ' ').replace('-', ' ').replace('"', ' ').replace("'", ' ').replace(',', ' ').lower().split()
        keyPhrases += [phrase]

    return keyPhrases

# <orderBy> remove a column -> wrong column name
def rule12(edits: dict) -> list:
    addArgs = edits['orderBy']['adds']
    rmArgs = edits['orderBy']['rms']

    adds = []
    rms = []

    for arg in addArgs:
        phrases = arg.lower().strip().split()
        for phrase in phrases:
            if not phrase in ruleWords:
                adds += [phrase]

    for arg in rmArgs:
        phrases = arg.lower().strip().split()
        for phrase in phrases:
            if not phrase in ruleWords:
                rms += [phrase]

    adds = set(adds)
    rms = set(rms)

    phrases = list(rms - adds)

    keyPhrases = []
    for phrase in phrases:
        phrase = phrase.replace('_', ' ').replace('-', ' ').replace('"', ' ').replace("'", ' ').replace(',', ' ').lower().split()
        keyPhrases += [phrase]

    return keyPhrases

# <SUBS> recursicvely apply aforementioned rules
