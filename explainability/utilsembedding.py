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
    if isinstance(corpus, str):
        corpus_tokens.append([t.lower_ for t in NLP(corpus) if (t.is_alpha and (t.lower_ not in STOPWORDS) and (len(t) >= MIN_LENGTH))])
            # print('from tokenizer::utilsembedding type(corpus_tokens) and len(corpus_tokens): ', type(corpus_tokens), len(corpus_tokens))
    else:
        for i, doc in enumerate(NLP.pipe(corpus)):
            if i % 1000 == 0:
                print(i, end=" ")
            corpus_tokens.append(
                [t.lower_ for t in doc if (t.is_alpha and (t.lower_ not in STOPWORDS) and (len(t) >= MIN_LENGTH))])            
    return corpus_tokens


def pretrained_models_gensim():
    print(list(gensim.downloader.info()['models'].keys()))


def save_model_gensim(checkpoint):
    model = gensim.downloader.load(checkpoint)
    model.save('../model/%s.d2v' % checkpoint)


def load_model_gensim(checkpoint):
    return KeyedVectors.load('../model/%s.d2v' % checkpoint)


def get_doc_embedding(model, tokens: List[str], dim=1): # called from get_docs_subspaces
    c = Counter(tokens)    
    words = [word for word in c.keys() if word in model.index_to_key]
    #print('from utilsembedding.get_doc_embedding, counter(tokens)= :', c, '\n no. of words=', len(words))
    if len(words) >= dim:
        doc_emb = np.array([model.get_vector(word) for word in words]).T
        freq_words = np.array([c[word] for word in words])
    elif len(words) == 0:
        doc_emb, freq_words = None, None
    else:
        num_new = dim-len(words)
        rand_words = list(np.random.choice(words,
                size=num_new, replace=True, ))
        words.extend([model.most_similar(word, topn=10)[np.random.randint(10)][0] for word in rand_words])
        # words.extend(rand_words)
        doc_emb = np.array([model.get_vector(word) for word in words]).T
        doc_emb += 0.0001 * np.random.randn(*doc_emb.shape)
        freq_words = np.ones(len(words))
    #print(freq_words)
    return doc_emb, freq_words


def get_doc_subspace(doc_emb, freq, dim): # called from get_docs_subspaces
    print('\n from utilsembedding.get_doc_subspace(), freq=', freq)
    U, _, _ = np.linalg.svd(doc_emb * np.sqrt(freq), full_matrices=False, compute_uv=True, hermitian=False)
    return U[:, :dim]


def get_docs_subspaces(corpus_tokens, model, emb_size=300, dim=20): # called from preprocessing
    print(f"subspace dimensionality is {dim}")
    out = np.zeros((len(corpus_tokens), emb_size, dim))
    for i, doc_tokens in enumerate(corpus_tokens):
        print('\n from get_docs_subspaces() type and len doc_tokens, and corpus_tokens =', type(doc_tokens), len(doc_tokens), 
              type(corpus_tokens),len(corpus_tokens))
        if i % 1000 == 0:
            print(i, end=" ")
        tmp, freq = get_doc_embedding(model, doc_tokens, dim)
        out[i] = get_doc_subspace(tmp, freq, dim=dim)
    return out

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


def preprocessing(corpus_text: List[str], subspace_dim, model_name='word2vec-google-news-300'):
    print("\n Tokenize the corpus!")    
    #print("\n Tokenized corpus", corpus_tokens)
    print("\n Loading the word embedding model!")
    model = load_model_gensim(model_name)
    print('\n Create the corpus subspaces!')
    if (isinstance(corpus_text, str)) | (len(corpus_text)==1):       
        corpus_tokens = tokenizer(corpus_text)
        c = Counter(corpus_tokens[0])        
        words = [word for word in c.keys() if word in model.index_to_key]
        assert len(words) >= MIN_LENGTH, f"the length of the text {len(words)} is lower than the dimensionality of the subspace {subspace_dim}"
        words_emb = np.array([model.get_vector(word) for word in words]).T
        freq_words = np.array([c[word] for word in words])
        F = np.diag(np.sqrt(freq_words))
        U, S, Vh = np.linalg.svd(words_emb @ F, full_matrices=False, compute_uv=True, hermitian=False)
        word_impact_no_rot = F @ Vh[:subspace_dim, :].T @ np.diag(1 / S[:subspace_dim])
        corpus_subspaces=U[:, :subspace_dim]
    else:
        corpus_tokens = tokenizer([corpus_text]) #added [0]
        corpus_subspaces = get_docs_subspaces(corpus_tokens, model, emb_size=EMBEDDING_DIM, dim=subspace_dim)
        
    return corpus_subspaces














# import transformers
# from transformers import AutoTokenizer, AutoModel
# word2vec: https://huggingface.co/fse/word2vec-google-news-300

# checkpoint = "bert-base-uncased" # 'fse/word2vec-google-news-300' # "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModel.from_pretrained(checkpoint)
# print(model)
