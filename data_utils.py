from torch.utils.data import Dataset
import os
import json
import torch
from transformers import RobertaTokenizer


def to_cuda(batch, gpuid):
    for n in batch:
        if n != "data":
            batch[n] = batch[n].to(gpuid)


def collate_mp(batch, pad_token_id, is_test=False):
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            result[i, :x.size(0)] = x
        return result
    
    src_input_ids = pad([x['src_input_ids'] for x in batch])
    
    # candidate
    candidate_ids = [x['candidate_ids'] for x in batch]
    max_len = max([max([len(c) for c in x]) for x in candidate_ids])
    candidate_ids = [pad(x, max_len) for x in candidate_ids]
    candidate_ids = torch.stack(candidate_ids)

    result = {
        'src_input_ids': src_input_ids,
        'candidate_ids': candidate_ids
    }

    if is_test:
        data = [x['data'] for x in batch]
        result['data'] = data
    else:   # train
        # positive weights
        pos_weights = torch.stack([x['positive_weights'] for x in batch])
        result['positive_weights'] = pos_weights

        # costs
        costs = torch.stack([x["costs"] for x in batch])
        result['costs'] = costs

        # negative
        negative_ids = [x['negative_ids'] for x in batch]
        max_len = max([max([len(c) for c in x]) for x in negative_ids])
        negative_ids = [pad(x, max_len) for x in negative_ids]
        result['negative_ids'] = torch.stack(negative_ids)
    return result
        


class SumDataset(Dataset):
    def __init__(self, fdir, model_type, max_len=-1, is_test=False, total_len=512, is_sorted=True, max_num=-1, is_untok=True, num=-1, neg_size=16, thre=0):
        """ dataformat : article, reference, [(candidate_i, score_i)]"""
        self.isdir = os.path.isdir(fdir)
        if self.isdir:
            self.fdir = fdir
            if num > 0:
                self.num = min(len(os.listdir(fdir)), num)
            else:
                self.num = len(os.listdir(fdir))
        else:
            with open(fdir) as f:
                self.files = [x.strip() for x in f]
            if num > 0:
                self.num = min(len(self.files), num)
            else:
                self.num = len(self.files)
        
        self.tok = RobertaTokenizer.from_pretrained(model_type, verbose=False)

        self.maxlen = max_len       # candidate max length
        self.maxnum = max_num       # candidate num
        self.is_test = is_test      # only evaluate
        self.total_len = total_len  # document max length
        self.sorted = is_sorted     
        self.is_untok = is_untok
        self.neg_size = neg_size    # negative num
        self.thre = thre


    def __len__(self):
        return self.num


    def __getitem__(self, idx):
        if self.isdir:
            with open(os.path.join(self.fdir, "%d.json"%idx), "r") as f:
                data = json.load(f)
        else:
            with open(self.files[idx]) as f:
                data = json.load(f)

        if self.is_untok:
            article = data['article_untok']
            abstract = data['abstract_untok']
        else:
            article = data['article']
            abstract = data['abstract']

        # document
        #src_txt = " ".join(article)
        cls_token = self.tok.cls_token
        src_txt = cls_token.join(article)
        src = self.tok.batch_encode_plus([src_txt], max_length=self.total_len, return_tensors='pt', pad_to_max_length=False, truncation=True)
        src_input_ids = src['input_ids']
        src_input_ids = src_input_ids.squeeze(0)

        # candidate 
        candidates = data['candidates_untok']
        _candidates = data['candidates']
        data['candidates'] = _candidates

        if self.sorted:     # Training
            candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
            _candidates = sorted(_candidates, key=lambda x:x[1], reverse=True)
            data['candidates'] = _candidates

        if self.maxnum > 0:
            candidates = candidates[:self.maxnum]
            _candidates = _candidates[:self.maxnum]
        
        if not self.is_untok:
            candidates = _candidates

        cand_txt = [" ".join(x[0]) for x in candidates]
        cand = self.tok.batch_encode_plus(cand_txt, max_length=self.maxlen, return_tensors='pt', pad_to_max_length=False, truncation=True, padding=True)
        candidate_ids = cand['input_ids']

        result = {
            'src_input_ids': src_input_ids,
            'candidate_ids': candidate_ids
        }

        if self.is_test:
            result['data'] = data
        else:   # train
            # positive weights
            pos_weights = data[str(self.thre)][:self.maxnum]
            result['positive_weights'] = torch.FloatTensor(pos_weights)

            # costs
            costs = torch.FloatTensor([(1-x[1]) for x in candidates])
            result['costs'] = costs

            # negative ids
            negatives = data['negative_untok']
            negatives = negatives[:self.neg_size]
            neg_txt = [" ".join(x) for x in negatives]

            neg = self.tok.batch_encode_plus(neg_txt, max_length=self.maxlen, return_tensors='pt', pad_to_max_length=False, truncation=True, padding=True)
            result['negative_ids'] = neg['input_ids']
        return result