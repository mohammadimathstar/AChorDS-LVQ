import numpy as np
import nltk
nltk.download('stopwords')
import gensim.downloader
from gensim.models import KeyedVectors

import spacy
from nltk.corpus import stopwords

from typing import List
from collections import Counter

# nltk.download('stopwords')
STOPWORDS = stopwords.words('english')

NLP = spacy.blank('en')
NLP.max_length= 4000000
EMBEDDING_DIM = 300
MIN_LENGTH = 3 # minimum number of character for a token
# NLP = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def tokenizer(corpus): #: List[str]): # called from preprocessing
    corpus_tokens = []
    #print('from tokenizer::utilsembedding type(corpus) and len(corpus)', type(corpus), len(corpus))
    corpus_tokens.append([t.lower_ for t in NLP(corpus) if (t.is_alpha and (t.lower_ not in STOPWORDS) and (len(t) >= MIN_LENGTH))])
            # print('from tokenizer::utilsembedding type(corpus_tokens) and len(corpus_tokens): ', type(corpus_tokens), len(corpus_tokens))         
    return corpus_tokens


def pretrained_models_gensim():
    print(list(gensim.downloader.info()['models'].keys()))


def save_model_gensim(checkpoint):
    model = gensim.downloader.load(checkpoint)
    model.save('../model/%s.d2v' % checkpoint)


def load_model_gensim(checkpoint):
    return KeyedVectors.load('../model/%s.d2v' % checkpoint)

def single_doc_embedding(tokens: List[str], dim):
    c = Counter(tokens[0])
    words = [word for word in c.keys() if word in model.index_to_key]
    assert len(words) >= dim, f"the length of the text {len(words)} is lower than the dimensionality of the subspace {dim}"
    words_emb = np.array([model.get_vector(word) for word in words]).T
    freq_words = np.array([c[word] for word in words])
    return words, words_emb, freq_words

def get_single_doc_subspaces(corpus_tokens, model, emb_size=300, dim=20): # called from preprocessing
    print("\n Compute the doc embedding from single doc: ", type(corpus_tokens), type(corpus_tokens[0]))
    words, words_emb, freq = single_doc_embedding(str(corpus_tokens), dim)
    F = np.diag(np.sqrt(freq))
    U, S, Vh = np.linalg.svd(words_emb @ F, full_matrices=False, compute_uv=True, hermitian=False)
    word_impact_no_rot = F @ Vh[:dim, :].T @ np.diag(1 / S[:dim])
    return U[:, :dim], words, words_emb, word_impact_no_rot


def preprocessing(corpus_text: str, subspace_dim, model_name='word2vec-google-news-300'):
    print("\n Tokenize the corpus!")   
    #print("\n Tokenized corpus", corpus_tokens)
    print("\n Loading the word embedding model!")
    model = load_model_gensim(model_name)
    print('\n Create the corpus subspaces!')
  #  corpus_tokens = tokenizer(corpus_text)
  #  corpus_subspaces=get_single_doc_subspaces(corpus_tokens, model, emb_size=EMBEDDING_DIM, dim=subspace_dim)
    #print('\n Length and type of corpus text from preprocessing', len(corpus_text), type(corpus_tokens), type([corpus_text]))
  #  if (isinstance(corpus_text, str)) | (len(corpus_text)==1):
    NLP = spacy.load("en_core_web_sm", exclude=["tok2vec", "parser", "ner", "attrbute_ruler"])     
    #corpus_tokens = tokenizer(corpus_text)
    #corpus_text=corpus_text[0]
    print(type(corpus_text), len(corpus_text), type(corpus_text[0]))
    corpus_tokens=[[word.lower_ for word in NLP(corpus_text) if (word.lower_ not in STOPWORDS) and (len(word) >= MIN_LENGTH)]]
    #print('\n type(corpus_tokens) : ', type(corpus_tokens),'type(doc_tokens): ', type(doc_tokens), 'len(corpus_tokens): ', len(corpus_tokens), 'len(doc_tokens): ', len(doc_tokens))
    c = Counter(corpus_tokens[0])        
    words = [word for word in c.keys() if word in model.index_to_key]
    assert len(words) >= MIN_LENGTH, f"the length of the text {len(words)} is lower than the dimensionality of the subspace {subspace_dim}"
    words_emb = np.array([model.get_vector(word) for word in words]).T
    freq_words = np.array([c[word] for word in words])
    #corpus_subspaces, words, words_emb, word_imp_no_rot = get_single_doc_subspaces(corpus_tokens, model, emb_size=EMBEDDING_DIM, 
    #                                                                               dim=subspace_dim)
    F = np.diag(np.sqrt(freq_words))
    U, S, Vh = np.linalg.svd(words_emb @ F, full_matrices=False, compute_uv=True, hermitian=False)
    word_impact_no_rot = F @ Vh[:subspace_dim, :].T @ np.diag(1 / S[:subspace_dim])
    corpus_subspaces=U[:, :subspace_dim]
        
    return corpus_subspaces














# import transformers
# from transformers import AutoTokenizer, AutoModel
# word2vec: https://huggingface.co/fse/word2vec-google-news-300

# checkpoint = "bert-base-uncased" # 'fse/word2vec-google-news-300' # "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModel.from_pretrained(checkpoint)
# print(model)
