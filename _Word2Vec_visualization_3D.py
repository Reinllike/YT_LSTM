# coding=UTF-8
import numpy as np
import matplotlib.pyplot as plt
import random
import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.decomposition import PCA
import imageio
import os
from github_getdata import getdata
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC']

os.mkdir(f"./gifs/")
## 0) Load the "word_sentence_list" for Word2Vec visualization
word_sentence_list = getdata('YT_output.csv')
model = gensim.models.KeyedVectors.load_word2vec_format('pre_trained.bin',
                                                        unicode_errors='ignore',
                                                        binary=True)
filenames = []
fig = plt.figure(figsize=(8, 8))
plt.ion()
ax = fig.gca(projection='3d')

def get_random_color():
    """获取一个随机的颜色"""
    r = lambda: random.uniform(0, 1)
    return [r(), r(), r(), 1]

## 1) 3D Visualization for Word2Vec
def display_pca_scatterplot(model, words, count):
    plt.cla()
    # Take word vectors
    word_vectors = np.array([model[w] for w in words])
    # PCA, take the first 2 principal components
    twodim = PCA().fit_transform(word_vectors)[:, :3]
    # list_num color
    color = get_random_color()
    # Draw
    ax.scatter(twodim[:, 0], twodim[:, 1], twodim[:, 2], edgecolors='k', c=color, s=300)
    for word, (x,y,z) in zip(words, twodim):
        ax.text(x+0.05, y+0.05, z+0.05, word, fontsize=24)
    ax.set_xlim(-75, 75)
    ax.set_ylim(-75, 75)
    ax.set_zlim(-75, 75)
    # plt.draw()
    plt.pause(0.1)

    ## 2) Save for the gif.
    filename = f'./gifs/{count}.png'
    filenames.append(filename)
    plt.savefig(filename)

count = 0
for words in word_sentence_list:
    count += 1
    print(words)
    display_pca_scatterplot(model, words, count)

## 3) Make gif from saved figures
with imageio.get_writer('word2Vec_gif.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

## 4) Remove the saved figures after making 'word2Vec_gif.gif'
for filename in set(filenames):
    os.remove(filename)

print("GIF Finished!")