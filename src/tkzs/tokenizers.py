import spacy
import torch
import re
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict


class ByteCharEncoder(object):
    
    def __init__(self):
        
        vocab = None
                
    def encode_token(self, token):
        """encode a single token: str"""
        return torch.LongTensor(list(token.encode("utf-8"))) + 1 # 0 for padding
    
    def encode_sequence(self, sequence):
        """encode a sequence of token: [str] or str"""
        return [self.encode_token(token) for token in sequence]
    
    def encode_batch(self, batch):
        """encode a batch: [[str]] or [str]"""
        
        lengths = []
        batch_tokens = []
        
        for seq in batch:
            batch_tokens.extend(self.encode_sequence(seq))
            lengths.append(len(seq))
            
        batch_tokens = pad_sequence(batch_tokens, batch_first=True)
            
        list_seq = torch.split(batch_tokens, lengths)
            
        return pad_sequence(list_seq, batch_first=True)
    
    def encode_batch_with_length(self, batch, length=15):
        """encode a batch: [[str]] or [str]"""
        
        lengths = []
        batch_tokens = []
        
        for seq in batch:
            batch_tokens.extend(self.encode_sequence(seq))
            lengths.append(len(seq))
        
        batch_tokens = torch.stack([self.pad_middle(i, length) for i in batch_tokens], dim=0)
        
        list_seq = torch.split(batch_tokens, lengths)
        
        return pad_sequence(list_seq, batch_first=True)
    
    def pad_middle(self, b, max_len):
        
        b = b[:max_len]
        
        len_b = len(b)
        
        pad_left = (max_len - len_b)//2
        
        if (len_b + max_len) % 2 == 0:
            return F.pad(b, (pad_left, pad_left))
        else:
            return F.pad(b, (pad_left + 1, pad_left))
        
    def pad_left(self, b, max_len):
        
        b = b[:max_len]
        
        len_b = len(b)
        
        pad_left = max_len - len_b
        
        return F.pad(b, (pad_left, 0))
    
    
class SpacyTokenizer(object):
    
    def __init__(self, name='en_core_web_sm'):
        
        self.spacy_model = spacy.load(name='en_core_web_sm', disable=['parser', 'ner'])
        
    def tokenize(self, sent):
        
        return [s.text for s in self.spacy_model(sent)]

def re_tokenizer(txt):
    """simple tokenization function based on regex

    Args:
        txt (str): string sequence

    Returns:
        list[str]: list of tokens
    """
    t = txt.strip().lower()
    t = re.sub(r'([%s])' % re.escape(string.punctuation), r' \1 ', t) 
    t = re.sub(r'\\.', r' ', t) 
    t = re.sub(r'\s+', r' ', t)
    return t.split()
    
    
class WordEncoder(object):
    
    def __init__(self, tokenizer=lambda x: x.split(), special_tokens=["[PAD]", "[UNK]"]):
        
        self.tokenizer = tokenizer
        
        self.special_tokens = special_tokens
        
        self.vocabulary = []
        
        self.token_to_id = {}
        
        self.id_to_token = {}
        
    def fit(self, documents):
        
        counter = defaultdict(int)
        
        for x in documents:
            x = self.check_type(x)
            for token in x:
                counter[token] += 1
                        
        self.vocabulary = self.special_tokens + [t for t in counter.keys()]
        
        for k, v in enumerate(self.vocabulary):
            self.token_to_id[v] = k
            self.id_to_token[k] = v
        
    def tokenize(self, x):
        """Tokenize a sequence"""
        
        return self.tokenizer(x)
    
    def batch_tokenize(self, doc):
        return [self.tokenizer(x) for x in doc]
    
    def encode(self, x):
        """encode sentence to list of ids"""
        
        x = self.check_type(x)
        y = []
        for t in x:
            try:
                idx = self.token_to_id[t]
            except KeyError:
                idx = 1
            y.append(idx)
            
        return y
    
    def encode_batch(self, doc, paddind_idx=0):
        """batch encode a list of sequence"""
        
        batch = []
        for b in doc:
            b = torch.LongTensor(self.encode(b))
            batch.append(b)
            
        return pad_sequence(batch, batch_first=True, padding_value=paddind_idx)
    
    def check_type(self, x):
        """if word already splitted, no need tokenization"""
        
        if isinstance(x, str):
            return self.tokenizer(x)
        elif isinstance(x, list):
            return x
        else:
            raise ValueError
            
    def __repr__(self):
        
        return f"tokenizer containing {len(self.vocabulary)} tokens"