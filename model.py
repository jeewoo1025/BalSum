import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaModel

from transformers.utils import logging
logger = logging.get_logger(__name__)


def NllLoss(score, labels):
    """
        link : https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        score [bsz, pos + all candidates] : cosine similarity
        labels [bsz]
    """
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(score, labels)
    return loss


def label_smoothed_NLL_loss(score, labels, epsilon=0.1):
    """
        link : https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/ResNet18_CIFAR10_Training_with_Input_Mixup_and_Label_Smoothing.ipynb
        score : [bsz, all candidates]
        labels : [bsz]
    """
    confidence = 1. - epsilon
    log_probs = F.log_softmax(score, dim=-1)  
    true_probs = torch.zeros_like(log_probs)
    true_probs.fill_(epsilon / (score.size(1) - 1))
    true_probs.scatter_(1, labels.unsqueeze(1), confidence)     
    loss = torch.mean(torch.sum(true_probs*-log_probs, dim=-1))
    return loss


def MultiNllLoss(score, neg_score, pos_weights=None, is_smoothing=True, device='cuda:0', is_IW=False):
    """
        Input:
            score : all positives (bsz, candi_num)
            neg_score : all negatives (bsz, neg_num)
    """
    if is_smoothing:
        loss_func = label_smoothed_NLL_loss
    else:
        loss_func = NllLoss
    TotalLoss = torch.FloatTensor([0]).to(device)

    if is_IW:
        # Add Masked & positive weights
        score = score*pos_weights

    n = score.size(1)   # positive num
    for i in range(n):
        pos_score = score[:, i].unsqueeze(-1)
        if pos_score.size(0) != neg_score.size(0):
            raise Exception(f'Batch size is wrong!!! pos_score ({pos_score.size(0)} != neg_score ({neg_score.size(0)})')

        total = torch.cat([pos_score, neg_score], 1)    # (bsz, pos + neg_num)
        label = torch.zeros(total.size(0)).long().to(device)
        
        loss = loss_func(total, label)
        TotalLoss += loss

    # divide by positive num
    TotalLoss = TotalLoss / n
    return TotalLoss


def MultiMarginLoss(costs, score, margin):
    """
        Input:
            costs : 1-(avg ROUGE score)
            score : similarity 
    """
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)   # 0.0

    # candidate loss
    n = score.size(1)
    for i in range(1, n):
        # positive 
        pos_score = score[:, :-i]
        pos_costs = costs[:, :-i]
        pos_score = pos_score.contiguous().view(-1)
        pos_costs = pos_costs.contiguous().view(-1)
        pos = pos_score + margin*pos_costs

        # negative
        neg_score = score[:, i:]
        neg_costs = costs[:, i:]
        neg_score = neg_score.contiguous().view(-1)
        neg_costs = neg_costs.contiguous().view(-1)
        neg = neg_score + margin*neg_costs

        ones = torch.ones_like(pos_score)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        loss = loss_func(pos, neg, ones)
        TotalLoss += loss
    return TotalLoss


class MLPLayer(nn.Module):
    """
        Head for getting sentence representations over RoBERTa's CLS representation
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


class Similarity(nn.Module):
    """
        Cosine similarity with temperature
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class BalSum(nn.Module):
    """
        Abstractive Summarization to Sentence Embeddings
    """
    def __init__(self, encoder, pad_token_id, cls_token_id, hidden_size, temp, gpuid):
        super(BalSum, self).__init__()

        self.encoder = RobertaModel.from_pretrained(encoder, cache_dir="./local_cache")
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.device = f'cuda:{gpuid}'


    def doc(self, input_id, batch_size):
        """
            get Document embeddings

            Input:
                - input_id : (bsz, seq_len)
            Output:
                - doc_emb : (bsz, K, dim)
                - cls_size : (bsz)
        """
        # get embeddings
        input_mask = input_id != self.pad_token_id
        outputs = self.encoder(input_id, attention_mask=input_mask)[0]      # (bsz, seq_len, dim)

        # get cls tokens
        cls_mask = input_id == self.cls_token_id
        K = torch.max(cls_mask.sum(-1)).item() # max cls token size

        cls_emb = None
        for bsz in range(batch_size):
            cur_output = outputs[bsz]
            cur_cls_ids = cls_mask[bsz].nonzero(as_tuple=True)[0].tolist()
            cur_cls_emb = cur_output[cur_cls_ids]

            # get current "CLS token size" && dimension(768)
            cur_size, cur_dim = cur_cls_emb.size(dim=0), cur_cls_emb.size(dim=1)    
            if cur_size < K:    # if current size < MAX CLS token size
                padding = torch.zeros((K-cur_size), cur_dim).to(self.device)
                cur_cls_emb = torch.cat([cur_cls_emb, padding], dim=0)
            cur_cls_emb = cur_cls_emb.unsqueeze(0)

            if bsz==0:
                cls_emb = cur_cls_emb
            else:
                cls_emb = torch.cat([cls_emb, cur_cls_emb], dim=0)
        return F.normalize(cls_emb, p=2, dim=2), cls_mask.sum(-1)


    def query(self, input_id, batch_size):
        """
            get summary embeddings

            Input:
                - input_id : (bsz, sum_num)
                - batch_size
            Output:
                - sum_emb : (bsz, sum_num, dim)
        """
        n = input_id.size(1)        # sum_num
        sum_id = input_id.view(-1, input_id.size(-1))   # (bsz*sum_num, dim)
        
        # get embeddings
        input_mask = sum_id != self.pad_token_id
        outputs = self.encoder(sum_id, attention_mask=input_mask)[0]      # (bsz*sum_num, seq_len, dim)
        
        # get CLS token
        sum_emb = outputs[:, 0, :]       # (bsz*sum_num, 1, dim)
        sum_emb = sum_emb.view(batch_size, n, -1)   # (bsz, sum_num, dim)
        return F.normalize(sum_emb, p=2, dim=2)


    def get_score(self, query_emb, doc_emb):
        """
            calculate Dot-product score between summaries and documents
            Input:
                - query_emb : (bsz, candi_size, dim)
                - doc_emb : (bsz, K, dim)
                - K : MAX CLS size
            Output:
                - score : (bsz, candi_size)
        """
        score = query_emb@doc_emb.permute(0, 2, 1)      # (bsz, candi_size, K)

        # Weighted Average
        K = doc_emb.shape[1]
        denom = torch.sum(score, dim=-1).unsqueeze(dim=-1).repeat_interleave(K, dim=-1)

        total_score = torch.sum(torch.div(torch.square(score), denom), dim=-1)
        return total_score        


    def forward(self, text_id, candidate_id=None, neg_id=None, is_test=False):
        """
            Calculating Sentence-Level Score
        """
        batch_size = text_id.size(0)

        # Document
        doc_emb, doc_cls_size = self.doc(text_id, batch_size)     # (bsz, MAX CLS size, dim)

        # Candidate
        candi_emb = self.query(candidate_id, batch_size)    # (bsz, candi_num, dim)

        # get similarity score using dot-product 
        score = self.get_score(candi_emb, doc_emb)

        if not is_test:
            # Negative Summaries
            neg_emb = self.query(neg_id, batch_size)

            # get similarity score using dot-product
            neg_score = self.get_score(neg_emb, doc_emb)

            return {'score': score, 'neg_score': neg_score}
        else:
            return {'score': score}