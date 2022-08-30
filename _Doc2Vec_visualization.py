# coding=UTF-8

import numpy as np
import matplotlib.pyplot as plt
import gensim
from gensim.models.doc2vec import  Doc2Vec
from github_getdata import getdata

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC']

word_sentence_list = getdata('YT_output.csv')

titles_list = word_sentence_list
print(len(titles_list))

# Load the pre_trained word2vec model: "pre_trained.bin"
model = gensim.models.KeyedVectors.load_word2vec_format('pre_trained.bin',
                                                        unicode_errors='ignore',
                                                        binary=True)

# Transfer Word2Vec to Doc2Vec by "Averaging Word embedding"
def document_vector(model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in model.key_to_index]
    return np.mean(model[doc], axis=0)

# Function that will help us drop documents that have no word vectors in word2vec
def has_vector_representation(word2vec_model, doc):
    """
    check if at least one word of the document is in the word2vec dictionary
    """
    return not all(word not in word2vec_model.key_to_index for word in doc)

# Filter out documents
def filter_docs(corpus, texts, condition_on_doc):
    """
    Filter corpus and texts given the function condition_on_doc which takes a doc. The document doc is kept if condition_on_doc(doc) is true.
    """
    number_of_docs = len(corpus)

    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]

    corpus = [doc for doc in corpus if condition_on_doc(doc)]
    # print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts)


# Preprocess the corpus
corpus = [title for title in titles_list]

# Remove docs that don't include any words in W2V's vocab
corpus, titles_list = filter_docs(corpus, titles_list, lambda doc: has_vector_representation(model, doc))

# Filter out any empty docs
corpus, titles_list = filter_docs(corpus, titles_list, lambda doc: (len(doc) != 0))
x = []
tt = 0
for doc in corpus:  # append the vector for each document
    x.append(document_vector(model, doc))
    # print(document_vector(model, doc).shape)
    if tt == 0:
        data_array = document_vector(model, doc)
    else:
        data_array = np.vstack( [data_array, document_vector(model, doc)] )
    tt += 1

# print( data_array.shape )
# print( data_array )

X = np.array(x)  # list to array
from sklearn.manifold import TSNE
import seaborn as sns
# Initialize t-SNE
tsne = TSNE(n_components=2, init='random', random_state=10, perplexity=100)

# Again use only 400 rows to shorten processing time
tsne_df = tsne.fit_transform(X[:len(word_sentence_list)])
tsne_df_1 = tsne.fit_transform(X[:247])
tsne_df_2 = tsne.fit_transform(X[247:2152])
tsne_df_3 = tsne.fit_transform(X[2152:2384])
tsne_df_4 = tsne.fit_transform(X[2384:len(word_sentence_list)])

fig, ax = plt.subplots(figsize=(14, 10))
# sns.scatterplot(tsne_df[:, 0], tsne_df[:, 1], alpha=0.5)
sns.scatterplot(tsne_df_1[:, 0], tsne_df_1[:, 1], alpha=0.5, hue=tsne_df_1[:, 1], palette="YlOrBr", legend = False)
sns.scatterplot(tsne_df_2[:, 0], tsne_df_2[:, 1], alpha=0.5, hue=tsne_df_2[:, 1], palette="BrBG", legend = False)
sns.scatterplot(tsne_df_3[:, 0], tsne_df_3[:, 1], alpha=0.5, hue=tsne_df_3[:, 1], palette="Blues", legend = False)
sns.scatterplot(tsne_df_4[:, 0], tsne_df_4[:, 1], alpha=0.5, hue=tsne_df_4[:, 1], palette="Greys", legend = False)

from adjustText import adjust_text
texts = []
titles_to_plot = list(np.arange(0, len(corpus), 60))  # plots every 60th title

# Append words to list
for title in titles_to_plot:
    title_Whole = ""
    for w in titles_list[title]:
        title_Whole = title_Whole + str(w)
    texts.append(plt.text(tsne_df[title, 0], tsne_df[title, 1], title_Whole, fontsize=10))
    # print(tsne_df[title, 0], tsne_df[title, 1], title_Whole)

# Plot text using adjust_text
adjust_text(texts, force_points=0.4, force_text=0.4,
            expand_points=(2, 1), expand_text=(1, 2),
            arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

plt.savefig('doc2Vec.jpg')
plt.show()