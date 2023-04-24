# !pip install open_clip_torch

import open_clip
# import tensorflow as tf
# import tensorflow_hub as hub
import torch
import gensim.downloader
from tqdm import tqdm

class USEEncode:
    def __init__(self, model=None, prompt=None):
        self.prefix, self.suffix = '', ''
        if prompt:
            self.prefix, self.suffix = prompt.split('{}')
        if model is None:
            model = "https://tfhub.dev/google/universal-sentence-encoder/4"
            
        self.model = hub.load(model)
    
    def encode(self, df):
        prompted = (self.prefix + df.color + self.suffix).values
        return torch.FloatTensor(self.model(prompted).numpy())

    
class CLIPEncode:
    def __init__(self, model=None, prompt=None):
        self.prefix, self.suffix = '', ''
        if prompt:
            self.prefix, self.suffix = prompt.split('{}')
        if model is None:
            model = "ViT-B-32"
        self.clip, _, _ = open_clip.create_model_and_transforms(
        model, pretrained="openai")
        
        self.tokenizer = open_clip.get_tokenizer(model)
    
    def encode(self, df):
        batch_size = 200
        prompted = (self.prefix + df.color + self.suffix).values
        with torch.no_grad():
            encodings = []
            for i in tqdm(range(0, len(prompted), batch_size)):
                tokenized = self.tokenizer(prompted[i:i+batch_size])
                encodings.append(self.clip.encode_text(tokenized))
        return torch.vstack(encodings)
    
    
class W2VEncode:
    def __init__(self, model=None, prompt=None):
        """
        Available Models:
         'fasttext-wiki-news-subwords-300',
         'conceptnet-numberbatch-17-06-300',
         'word2vec-ruscorpora-300',
         'word2vec-google-news-300',
         'glove-wiki-gigaword-50',
         'glove-wiki-gigaword-100',
         'glove-wiki-gigaword-200',
         'glove-wiki-gigaword-300',
         'glove-twitter-25',
         'glove-twitter-50',
         'glove-twitter-100',
         'glove-twitter-200',
        """
        if model is None:
            model = 'glove-wiki-gigaword-50'
        self.vectorizer = gensim.downloader.load(model)
    
    def encode(self, df):
        return torch.FloatTensor(self.vectorizer[df.color.values])