import numpy as np
import pandas as pd
from utils_model import *
from utilsembedding import *
from wordcloud import WordCloud

def return_model(fname):
    with np.load(fname + '.npz', allow_pickle=True) as f:
        xprotos, yprotos = f['xprotos'], f['yprotos']
        lamda = f['lamda']
        print(f"train accuracy: {f['accuracy_of_train_set'][-1]}, "
              f"\t validation accuracy: {f['accuracy_of_validation_set'][-1]} ({np.max(f['accuracy_of_validation_set'])})")

        if 'conf_mat' in f.keys():
            print(f['conf_mat'])
    return xprotos, yprotos, lamda

def get_example_txt(idx: int, cls: int, label_col='label', fname: str='reuters-8-test.cvs'):

    df = pd.read_csv(fname)
    return df.loc[df[label_col] == cls, 'text'].tolist()[idx], df.loc[df[label_col] == cls, label_col].tolist()[idx]


def doc_embedding(tokens: List[str], dim):
    c = Counter(tokens)
    words = [word for word in c.keys() if word in MODEL.index_to_key]
    assert len(words) >= dim, f"the length of the text {len(words)} is lower than the dimensionality of the subspace {dim}"

    words_emb = np.array([MODEL.get_vector(word) for word in words]).T
    freq_words = np.array([c[word] for word in words])

    return words, words_emb, freq_words

def get_subspace_of_doc(txt, dim=1):
    print("Tokenize the corpus!")
    doc_tokens = tokenizer([txt])[0]

    print("Compute the doc embedding")
    words, words_emb, freq = doc_embedding(doc_tokens, dim)

    F = np.diag(np.sqrt(freq))
    U, S, Vh = np.linalg.svd(words_emb @ F, full_matrices=False, compute_uv=True, hermitian=False)

    word_impact_no_rot = F @ Vh[:dim, :].T @ np.diag(1 / S[:dim])
    return U[:, :dim], words, words_emb, word_impact_no_rot

def get_winner_prototypes(subspace_doc, label, xprotos, yprotos, rel, metric_type='geodesic'):
    plus, minus = nearest_prototypes(subspace_doc, label, xprotos, yprotos, metric_type, relevance=rel)
    return plus, minus

def get_words_importance_all_dir(word_impact_no_rot, result: dict, rel: np.array):
    """returns a (n x D) matrix: each row captures the importance of a word"""
    return word_impact_no_rot @ result['Q'] @ np.diag(rel[0]) #changed

def get_most_similar_word(vec, topn=10):
    tmp = {
        k: 100 * round(v, 3) for (k, v) in MODEL.most_similar(positive=[vec], negative=[], topn=topn)
    }
    return tmp

def create_word_cloud(d, background_color='white'):
    wc = WordCloud(background_color=background_color)
    minD = np.min([n for n in d.values()])
    if minD<=0:
        d = {w: f - minD + 1 for w, f in d.items()}
    wc.generate_from_frequencies(d)
    return wc

def plot_word_cloud(text, title="", fname="tmp", background_color='white'):

    cloud = create_word_cloud(text, background_color=background_color) #for text in wc_texts]

    fig, axes = plt.subplots(1, 1, figsize=(4, 2.5))
    _ = axes.imshow(cloud, interpolation='bilinear')

    _ = axes.grid(False)
    _ = axes.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    if True:
        plt.savefig("../pics/top_words_%s.eps" % fname, dpi=150)
    plt.show()

def main(txt, label, dim, xprotos, yprotos, rel, weight_type='abs', metric_type='geodesic', dic=None):
    subspace, words, words_emb, word_imp_no_rot = get_subspace_of_doc(txt, dim)
    plus, minus = get_winner_prototypes(subspace, label, xprotos, yprotos, rel, metric_type)

    if plus['distance'] > minus['distance']:
        t = 'misclassified'
    else:
        t = 'correct'

    print(
        f"""\n\t\t{t}: \t  {plus['distance']-minus['distance']}, 
        normalized:    {(plus['distance']-minus['distance'])/(plus['distance']+minus['distance'])}"""
    )

    winners_type = ["positive", 'negative']
    winners = [plus, minus]
    word_importances_winners = []

    for wtype, w in zip(winners_type, winners):
        if dic is not None:
            title = f"{wtype}: {dic[yprotos[w['index']]]}"
        else:
            title = f"{wtype}: {yprotos[w['index']]}"
        print(title)

        X, M = words_emb, word_imp_no_rot
        W = X.T @ xprotos[w['index']] @ w['Qw']
        if rel.shape[0] == 1:
            txt_words_impact = rel * M * W
        else:
            txt_words_impact = np.expand_dims(rel[w['index']], axis=0) * M * W

        if weight_type == 'abs':
            txt_words_impact = np.abs(txt_words_impact)

        word_importances_winners.append(txt_words_impact)

        weights_sort_idx = np.argsort(txt_words_impact, axis=None)
        txt_words_impact = txt_words_impact.flatten()

        tmp = {words[i // subspace.shape[-1]]: txt_words_impact[i] for i in weights_sort_idx[-1:-num_of_words_text:-1]}
        print(f"text : {tmp}")

        plot_word_cloud(
            tmp,
            # topn=num_of_words_text,
            title=title,
            fname="_text_%s_%s_global" % (dataname, wtype),
            background_color='white'  # 'lightgrey'
        )

    word_importance_diff = word_importances_winners[0] - word_importances_winners[1]
    # word_importance_diff = np.abs(word_importances_winners[0]) - np.abs(word_importances_winners[1])
    sort_idx = np.argsort(word_importance_diff, axis=None)
    words_impacts = word_importance_diff.flatten()

    tmp = {words[i // subspace.shape[-1]]: words_impacts[i] for i in sort_idx[-1:-num_of_words_text:-1]}
    print(f"diff : {tmp}")

    plot_word_cloud(
        tmp,
        # topn=num_of_words_text,
        # title='differences',
        fname="_text_%s_global_diff_%i" % (dataname, idx),
        background_color='white'  # 'lightgrey'
    )

    # tmp = {words[i // subspace.shape[-1]]: txt_words_impact[i] for i in weights_sort_idx[-1:-num_of_words_text:-1]}
    # print(f"text : {tmp}")

    print("relevance:", rel)

# TODO: write a code that compute closest words to the subspaces generated by
#  prototypes (for prototypes' interpretations')
#dataname = #'hyperpartisan' # 'reuters-8' , 'newsgroups20', 'arxiv-4', 'housing', hyperpartisan
dataname='reuters-8'
# fname = "../model/%s/model_d20_ps" % dataname
# fname = "../model/eviction/eviction_model_d30_ps"
# fname = "../model/%s/glove/%s_model_d20_ps" % (dataname, dataname)
# fname = "../model/%s/transformer/model_d20_ge" % (dataname)
#fname = "../model/hyperpartisan/glove/hyperpartisan_model_d30_ps_92.31"
fname='../model/reuters-8/glove/reuters-8_model_d20_ps'

num_of_words_text = 20 # number of top words to show
cls = 1
idx = 0

# data_name = '../data/%s-test.csv' % dataname
# data_name = '../data/Long-Document/longdocs4class.csv'
# data_name = '../data/housing/both_annotations_with_texts.csv'
data_name = '../data/%s/%s-test.csv' % (dataname, dataname) #imdb

weight_type = 'nonabs', # 'abs' or 'nonabs'

# misclassified: 
# misclassified: (reutrers) cls=1, idx=1,      idx=0, cls=4
print('Loading a text')
txt, label = get_example_txt(
    idx=idx, cls=cls,
    # label_col='Housing',# Housing, Exit
    fname=data_name
)
print(txt, "\n")
print(f"There are {len(txt.split())} tokens. \n")


if dataname == 'reuters-8':
    dic = {
        'acq':0, 'crude': 1, 'earn': 2, 'grain': 3,
        'interest': 4, 'money-fx': 5, 'ship': 6, 'trade': 7
    }
elif dataname == 'newsgroups20':
    dic = {
        'alt.atheism': 0, 'comp.graphics': 1,
        'comp.os.ms-windows.misc': 2,
        'comp.sys.ibm.pc.hardware': 3,
        'comp.sys.mac.hardware': 4, 'comp.windows.x': 5,
        'misc.forsale': 6, 'rec.autos': 7,
        'rec.motorcycles': 8, 'rec.sport.baseball': 9,
        'rec.sport.hockey': 10, 'sci.crypt': 11,
        'sci.electronics': 12, 'sci.med': 13, 'sci.space': 14,
        'soc.religion.christian': 15, 'talk.politics.guns': 16,
        'talk.politics.mideast': 17, 'talk.politics.misc': 18,
        'talk.religion.misc': 19
    }
elif dataname == 'arxiv-4':
    dic = {'cs.IT': 0, 'cs.NE': 1, 'math.AC': 2, 'math.GR': 3}
elif dataname == 'arxiv-11':
    dic = {
    'cs.AI': 0, 'cs.CE': 1, 'cs.CV': 2, 'cs.DS': 3,
    'cs.IT': 4, 'cs.NE': 5, 'cs.PL': 6, 'cs.SY': 7,
    'math.AC': 8, 'math.GR': 9, 'math.ST': 10
    }
elif dataname == 'housing':
    dic = {
        'housing': 1,
        'non-housing': 0,
    }
else:
    dic = None

if dic is not None:
    dic = {v: k for k, v in dic.items()}

print("Loading the GRLGQ model!")
xprotos, yprotos, rel = return_model(fname)
dim = xprotos.shape[-1]
D = xprotos.shape[-2]

print("Loading the word embedding model!")
if D == 100:
    checkpoint = 'glove.6B.100d', #'word2vec-google-news-300', 'glove.6B.100d'
elif D == 300:
    checkpoint = 'glove.42B.300d'# 'word2vec-google-news-300'
else:
    print("give a model")
    checkpoint = None

MODEL = load_model_gensim(
    checkpoint=checkpoint
)

main(txt, label, dim, xprotos, yprotos, rel, weight_type='nonabs', metric_type='geodesic', dic=dic)
