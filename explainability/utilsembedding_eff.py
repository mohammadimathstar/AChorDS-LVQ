import numpy as np

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



def pretrained_models_gensim():
    print(list(gensim.downloader.info()['models'].keys()))


def save_model_gensim(checkpoint):
    model = gensim.downloader.load(checkpoint)
    model.save('../model/%s.d2v' % checkpoint)


def load_model_gensim(checkpoint):
    return KeyedVectors.load('../model/%s.d2v' % checkpoint)


def get_doc_embedding(model, tokens: List[str], dim):
    c = Counter(tokens)
    words = [word for word in c.keys() if word in model.index_to_key]
    if len(words) >= dim:
        doc_emb = np.array([model.get_vector(word) for word in words]).T
        freq_words = np.array([c[word] for word in words])
    elif len(words) == 0:
        doc_emb, freq_words = None, None
    else:
        num_new = dim-len(words)
        rand_words = list(
            np.random.choice(
                words,
                size=num_new,
                replace=True,
            )
        )
        freq_words = [c[word] for word in words]
        words.extend([model.most_similar(word, topn=10)[np.random.randint(10)][0] for word in rand_words])
        doc_emb = np.array([model.get_vector(word) for word in words]).T
        # doc_emb = np.concatenate((doc_emb, np.zeros((doc_emb.shape[0], num_new))), axis=1)

        doc_emb += 0.00001 * np.random.randn(*doc_emb.shape)
        freq_words.extend([1] * num_new)
        freq_words = np.array(freq_words)
        # freq_words = np.ones(doc_emb.shape[1])
        # freq_words = np.ones(len(words))

    return doc_emb, freq_words


def get_doc_subspace(doc_emb, freq, dim=MIN_LENGTH, sing_values=False):
    U, S, _ = np.linalg.svd(doc_emb * np.sqrt(freq), full_matrices=False, compute_uv=True, hermitian=False)
    if sing_values:
        return U[:, :dim], S
    else:
        return U[:, :dim], None


def get_docs_subspaces(corpus_tokens, model, emb_size=300, dim=MIN_LENGTH, weighted=True, sing_values=False):
    if sing_values:
        sings = np.zeros((len(corpus_tokens), dim))
    else:
        sings = None
    out = np.zeros((len(corpus_tokens), emb_size, dim))
    for i, doc_tokens in enumerate(corpus_tokens):
        tmp, freq = get_doc_embedding(model, doc_tokens, dim)
        if not weighted:
            freq = 1
        if freq is None:
            return None, None
        out[i], s = get_doc_subspace(tmp, freq, dim=dim, sing_values=sing_values)
        if sing_values:
            sings[i] = s
    return out, sings


def preprocessing(corpus_text: List[str], subspace_dim, weighted=True, model_name='word2vec-google-news-300', sing_values=False):

    print("\nLoading the word embedding model!")
    model = load_model_gensim(model_name)

    if sing_values:
        sings = np.zeros((len(corpus_text), subspace_dim))
    else:
        sings = None

    corpus_subspaces = np.zeros((len(corpus_text), EMBEDDING_DIM, subspace_dim))
    start_idx = 0
    for i, doc in enumerate(NLP.pipe(corpus_text[start_idx:])):

        if i % 500 == 0:
            print(start_idx + i, end=" ")
        doc_tokens = [
            t.lower_ for t in doc if (t.is_alpha and (t.lower_ not in STOPWORDS) and (len(t) >= MIN_LENGTH))
        ]

        sub, s = get_docs_subspaces(
            [doc_tokens],
            model,
            weighted=weighted,
            emb_size=EMBEDDING_DIM,
            dim=subspace_dim,
            sing_values=sing_values
        )

        if corpus_subspaces[start_idx+i] is not None:
            corpus_subspaces[start_idx + i] = sub
        else:
            print(f"\n{start_idx+i}-th doc does not have subspace (no valid words)\n")
        if sing_values:
            sings[start_idx+i] = s[0]

        if (i+1) % 500 == 0:
            np.savez('../data/%s_emb%i_train.npz' % ('tmp', start_idx), X=corpus_subspaces)

    return corpus_subspaces, sings


if __name__ == '__main__':
    checkpoint = 'glove-wiki-gigaword-200'
    save_model_gensim(checkpoint)

# import transformers
# from transformers import AutoTokenizer, AutoModel
# word2vec: https://huggingface.co/fse/word2vec-google-news-300

# checkpoint = "bert-base-uncased" # 'fse/word2vec-google-news-300' # "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModel.from_pretrained(checkpoint)
# print(model)