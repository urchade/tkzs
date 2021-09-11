# tkzs

## Installation

```bash
pip install tkzs
```

## Use a simple tokenizer

```python
from tkzs.tokenizers import re_tokenizer

txt = "Contrastive Fine-tuning Improves Robustness for Neural Rankers"

re_tokenizer(txt)
```

## Use a spacy word tokenizer

```python
from tkzs.tokenizers import SpacyTokenizer

txt = "Contrastive Fine-tuning Improves Robustness for Neural Rankers"

tokenizer = SpacyTokenizer(name='en_core_web_sm')

tokenizer.tokenize(txt)
```

## Use a word encoder

```python
from tkzs.encoders import WordEncoder
from tkzs.tokenizers import re_tokenizer

docs = [
    "Contrastive Fine-tuning Improves Robustness for Neural Rankers",
    "Unsupervised Neural Machine Translation for Low-Resource Domains via Meta-Learning",
    "Spatial Dependency Parsing for Semi-Structured Document Information Extraction"
    ]

encoder = WordEncoder(tokenizer=re_tokenizer)

encoder.fit(docs)

encoder.batch_tokenize(docs) # return a list of tokenized sequence

encoder.encode_batch(docs) # return a tensor of size [batch_size, max_length]
```

## Use a byte encoder

```python
from tkzs.encoders import ByteEncoder
from tkzs.tokenizers import re_tokenizer

docs = [
    "Contrastive Fine-tuning Improves Robustness for Neural Rankers",
    "Unsupervised Neural Machine Translation for Low-Resource Domains via Meta-Learning",
    "Spatial Dependency Parsing for Semi-Structured Document Information Extraction"
    ]

encoder = ByteEncoder()

# return a tensor of shape [Batch, Word, Char]
encoder.encode_batch(docs, char_padding='center', word_length=None, tokenizer=re_tokenizer)

```
