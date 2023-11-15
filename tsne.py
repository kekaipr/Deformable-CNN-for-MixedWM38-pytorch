import torch 
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
import os
import umap
import cv2
from sklearn import manifold, datasets
# defect = 'random'
# defect_dict = {1:[1, 0, 1, 0, 0, 0, 0, 0],
#                2:[1, 0, 1, 0, 1, 0, 0, 0],
#                3:[1, 0, 1, 0, 0, 0, 1, 0],
#                4:[1, 0, 0, 1, 0, 0, 0, 0],
#                5:[1, 0, 0, 1, 1, 0, 0, 0],
#                6:[1, 0, 0, 1, 0, 0, 1, 0],
#                7:[1, 0, 0, 0, 1, 0, 0, 0],
#                8:[1, 0, 1, 0, 1, 0, 1, 0],
#                9:[1, 0, 0, 1, 1, 0, 1, 0],
#                10:[1, 0, 0, 0, 1, 0, 1, 0],
#                11:[1, 0, 0, 0, 0, 0, 1, 0],
#                12:[0, 1, 1, 0, 0, 0, 0, 0],
#                13:[0, 1, 1, 0, 1, 0, 0, 0],
#                14:[0, 1, 1, 0, 0, 0, 1, 0],
#                15:[0, 1, 0, 1, 0, 0, 0, 0],
#                16:[0, 1, 0, 1, 1, 0, 0, 0],
#                17:[0, 1, 0, 1, 0, 0, 1, 0],
#                18:[0, 1, 0, 0, 1, 0, 0, 0],
#                19:[0, 1, 1, 0, 1, 0, 1, 0],
#                20:[0, 1, 0, 1, 1, 0, 1, 0],
#                21:[0, 1, 0, 0, 1, 0, 1, 0],
#               22:[0, 1, 0, 0, 0, 0, 1, 0],
#                23:[0, 0, 1, 0, 1, 0, 0, 0],
#                24:[0, 0, 1, 0, 1, 0, 1, 0],
#                25:[0, 0, 1, 0, 0, 0, 1, 0],
#                26:[0, 0, 0, 1, 1, 0, 0, 0],
#                27:[0, 0, 0, 1, 1, 0, 1, 0],
#                28:[0, 0, 0, 1, 0, 0, 1, 0],
#                29:[0, 0, 0, 0, 1, 0, 1, 0],
#                30: [1, 0, 0, 0, 0, 0, 0, 0],
#                31:  [0, 1, 0, 0, 0, 0, 0, 0],
#                32: [0, 0, 1, 0, 0, 0, 0, 0],
#                33: [0, 0, 0, 1, 0, 0, 0, 0],
#                34:[0, 0, 0, 0, 1, 0, 0, 0],
#                35:[0, 0, 0, 0, 0, 1, 0, 0],
#                36: [0, 0, 0, 0, 0, 0, 0, 0], 
#                37:[0, 0, 0, 0, 0, 0, 0, 1],
#                38:[0, 0, 0, 0, 0, 0, 1, 0], 
#                }
# def find_key_by_value(dictionary, value):
#     for key, val in dictionary.items():
#         if val == value:
#             return key

defect_list = ['none', 'center', 'donut', 'edge_LOC', 'edge_ring', 'LOC', 'nearfull', 'scratch', 'random']

plt.figure(figsize=(15, 15))
plt.rcParams["font.family"] = "Times New Roman"

for i, defect in enumerate(defect_list, 1):
    path38 = os.path.join(os.getcwd(), f'datasets/for_CAE/for_CAE_38_{defect}.npz')
    path811 = os.path.join(os.getcwd(), f'datasets/for_CAE/for_CAE_811_{defect}.npz')
    mix38 = np.load(path38)
    wm811 = np.load(path811)

    sample_num = mix38['arr_0'].shape[0] + wm811['arr_0'].shape[0]
    sample = np.concatenate((mix38['arr_0'], wm811['arr_0']), axis=0)

    label_38, label_811 = label = np.zeros((mix38['arr_0'].shape[0]), dtype=np.uint8), np.ones((wm811['arr_0'].shape[0]), dtype=np.uint8)

    label = np.append(label_38, label_811)

    im = np.zeros((sample_num, 52 * 52), dtype=np.uint8)
    im = sample.reshape(sample_num, 52 * 52)
    im = im.astype(float)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    embedding = tsne.fit_transform(im)

    # reducer = umap.UMAP()
    # embedding = reducer.fit_transform(im)

    plt.subplot(3, 3, i)
    plt.title(f't-SNE projection of {defect} in two datasets')
    plt.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(3) - 0.5).set_ticks(np.arange(2))

plt.tight_layout()
plt.show() 



