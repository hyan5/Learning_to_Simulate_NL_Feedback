import re
import pdb


TAGS = ['from', 'where', 'select', 'except', 'limit', 'groupBy', 'orderBy', 'having', 'intersect', 'union']

class EditsParser():
    def find_match(self, pattern, inp):
        adds = []
        rms = []
        for match in re.findall(pattern, inp):
            mat = match.strip().split()
            if mat[0] == 'add':
                adds.append(' '.join(mat[1:]))
            if mat[0] == 'remove':
                rms.append(' '.join(mat[1:]))

        redundant = list(set(adds) & set(rms))
        for arg in redundant:
            adds.remove(arg)
            rms.remove(arg)

        res = {'adds': list(set(adds)), 'rms': list(set(rms))}
        return res

    def from_parser(self, inp):
        pattern = r'< from >(.*?)</ from >'
        return self.find_match(pattern, inp)

    def select_parser(self, inp):
        pattern = r'< select >(.*?)</ select >'
        return self.find_match(pattern, inp)

    def where_parser(self, inp):
        pattern = r'< where >(.*?)</ where >'
        return self.find_match(pattern, inp)

    def except_parser(self, inp):
        pattern = r'< except >(.*?)</ except >'
        return self.find_match(pattern, inp)

    def limit_parser(self, inp):
        pattern = r'< limit >(.*?)</ limit >'
        return self.find_match(pattern, inp)

    def groupBy_parser(self, inp):
        pattern = r'< groupBy >(.*?)</ groupBy >'
        return self.find_match(pattern, inp)

    def orderBy_parser(self, inp):
        pattern = r'< orderBy >(.*?)</ orderBy >'
        return self.find_match(pattern, inp)

    def having_parser(self, inp):
        pattern = r'< having >(.*?)</ having >'
        return self.find_match(pattern, inp)

    def intersect_parser(self, inp):
        pattern = r'< intersect >(.*?)</intersect >'
        return self.find_match(pattern, inp)

    def union_parser(self, inp):
        pattern = r'< union >(.*?)</ union >'
        return self.find_match(pattern, inp)
    
    def parse_edits(self, inp):
        parse_dict = {}
        for tag in TAGS:
            if tag == 'select':
                parse_dict[tag] = self.select_parser(inp)
            if tag == 'from':
                parse_dict[tag] = self.from_parser(inp)
            if tag == 'where':
                parse_dict[tag] = self.where_parser(inp)
            if tag == 'except':
                parse_dict[tag] = self.except_parser(inp)
            if tag == 'limit':
                parse_dict[tag] = self.limit_parser(inp)
            if tag == 'groupBy':
                parse_dict[tag] = self.groupBy_parser(inp)
            if tag == 'orderBy':
                parse_dict[tag] = self.orderBy_parser(inp)
            if tag == 'having':
                parse_dict[tag] = self.having_parser(inp)
            if tag == 'intersect':
                parse_dict[tag] = self.intersect_parser(inp)
            if tag == 'union':
                parse_dict[tag] = self.union_parser(inp)
        return parse_dict

class EditsEval():
    def __init__(self, preds, refs):
        self.preds = preds
        self.refs = refs
        self.edits_parser = EditsParser()
    
    def compare_edit(self, pred, ref):
        if len(pred) != len(ref):
            return False
        ref_len = len(ref)
        can_list = pred + ref
        return ref_len == len(list(set(can_list)))

    def exact_match(self, pred, ref):
        for tag in TAGS:
            pre = pred[tag]
            tar = ref[tag]
            if not self.compare_edit(pre['adds'], tar['adds']) or not self.compare_edit(pre['rms'], tar['rms']):
                return False
        return True

    def cal_progress(self, pred, ref):
        initial_edits_count = 0
        pred_edits_count = 0
        for tag in TAGS:
            for add in ref[tag]['adds']:
                initial_edits_count += 1
                if add not in pred[tag]['adds']:
                    pred_edits_count += 1
            for rm in ref[tag]['rms']:
                initial_edits_count += 1
                if rm not in pred[tag]['rms']:
                    pred_edits_count += 1
            for add in pred[tag]['adds']:
                if add not in ref[tag]['adds']:
                    pred_edits_count += 1
            for rm in pred[tag]['rms']:
                if rm not in ref[tag]['rms']:
                    pred_edits_count += 1
        return 1. * (initial_edits_count - pred_edits_count) / initial_edits_count

    def evaluation(self):
        num_total = 0 
        num_correct = 0
        total_progress = 0
        num_edits_decreased = 0
        num_edits_increased = 0

        for pred, ref in zip(self.preds, self.refs):
            # print(f'Gold: {ref}')
            # print(f'Pred: {pred}')

            num_total += 1
            pred_dict = self.edits_parser.parse_edits(pred)
            ref_dict = self.edits_parser.parse_edits(ref)
            if self.exact_match(pred_dict, ref_dict):
                num_correct += 1
            progress = self.cal_progress(pred_dict, ref_dict)
            total_progress += progress
            if progress < 0:
                num_edits_increased += 1
            elif progress > 0:
                num_edits_decreased += 1
        exact_match_score = 1. * num_correct / num_total
        progress_score = 1. * total_progress / num_total
        edits_increased = 1. * num_edits_increased / num_total
        edits_decreased = 1. * num_edits_decreased / num_total
        return {'exact_match': exact_match_score, 'progress': progress_score, 'edits_increased': edits_increased, 'edits_decreased': edits_decreased}

    
if __name__=='__main__':
    preds = []
    refs = []
    with open('error_analysis/splash/feqs_test_edits.sim', 'r') as f:
        preds = [line.strip() for line in f.readlines()]
    with open('error_analysis/test.target', 'r') as f:
        refs = [line.strip() for line in f.readlines()]

    # preds = ['< from > add store </ from > < from > remove district </ from >']
    # refs = ['< from > add store </ from > < from > add stroe district </ from >']
    edit_eval = EditsEval(preds, refs)

    results = edit_eval.evaluation()
    print(f'Dev Evaluation Result: {results}')

    # with open('error_analysis/feqs_train.sim', 'r') as f:
        # preds = [line.strip() for line in f.readlines()]
    # with open('error_analysis/test.target', 'r') as f:
        # refs = [line.strip() for line in f.readlines()]

    # preds = ['< from > add store </ from > < from > remove district </ from >']
    # refs = ['< from > add store </ from > < from > add stroe district </ from >']
    # edit_eval = EditsEval(preds, refs)

    # results = edit_eval.evaluation()
    # print(f'Test Evaluation Result: {results}')