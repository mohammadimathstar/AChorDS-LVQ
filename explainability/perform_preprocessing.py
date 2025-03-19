import pandas as pd
from utilsembedding_eff import *

def do_preprocessing(
        dataname,
        dim,
        weighted=True,
        model_name='word2vec-google-news-300',
        # num_words_per_doc=10, set_size=50
):

    print(dataname)
    
    if dataname == 'reuters-8':
        df_tr = pd.read_csv('../data/%s-train-all-terms.csv' % dataname)
        df_te = pd.read_csv('../data/%s-test-all-terms.csv' % dataname)
        corpus_tr, ytrain = df_tr['text'].tolist(), df_tr['label'].tolist()
        corpus_val, yval = df_te['text'].tolist(), df_te['label'].tolist()
    elif dataname == 'arxiv-4':
        df_tr = pd.read_csv('../data/Long-Document/longdocs4class.csv')
        corpus_tr, ytrain = df_tr['text'].tolist(), df_tr['label'].tolist()
    elif dataname == 'hyperpartisan':
        df_tr = pd.read_csv('../data/%s/%s-train.csv' % (dataname, dataname))
        df_te = pd.read_csv('../data/%s/%s-test.csv' % (dataname, dataname))
        # df_val = pd.read_csv('../data/%s/%s-dev.csv' % (dataname, dataname))
        # df_te = pd.concat([df_te, df_val], axis=0)
        corpus_tr, ytrain = df_tr['text'].tolist(), df_tr['label'].tolist()
        corpus_val, yval = df_te['text'].tolist(), df_te['label'].tolist()
    else:
        raise Exception("Sorry, the data set is not available")

    # ****************************
    # ****** Word Embedding ******
    # ****************************
    print('\nPreprocessing training set ...')
    print(f"There are {len(corpus_tr)} training examples!")
    xtrain, sing_val = preprocessing(
        corpus_tr,
        subspace_dim=dim,
        weighted=weighted,
        model_name=model_name,
        sing_values=False
    )
    np.savez('../data/%s/emb%i_train.npz' % (dataname, dim), xtrain=xtrain, ytrain=ytrain, singular_vals=sing_val)
    del xtrain

    print('\n\nPreprocessing testing set ...')
    print(f"There are {len(corpus_val)} testing examples!")
    xval, _ = preprocessing(
        corpus_val,
        subspace_dim=dim,
        weighted=weighted,
        model_name=model_name,
        sing_values=False
    )
    np.savez('../data/%s/emb%i_test.npz' % (dataname, dim), xtest=xval, ytest=yval)


if __name__ == '__main__':
    # datanames:
    # 'arxiv-11' #'reuters-8', 'newsgroups20', housing,
    # article8, P1-1, housing-not-annotated, imdb, movie-review
    dataname = 'hyperpartisan'# 'article-8'

    do_preprocessing(
        dataname=dataname,
        dim=50,
        # weighted=True,
        model_name='glove.42B.300d'
        # 'glove.42B.300d',
        #'word2vec-google-news-300', 'glove.6B.100d',
        # glove-wiki-gigaword-200, glove-wiki-gigaword-300, glove.42B.300d
    )

