"""
By default, the OpenAI embeddings are pretty huge: dim=1536.

This script uses PCA to reduce the dimensionality of the embeddings to any size of your choosing
and saves them to a new JSON file "embeddings_{dim}.json".

Like "embeddings.json", this new file maps from text -> embedding.
"""


import json
import numpy as np
from sklearn.decomposition import PCA

front = "The rectus muscles have lines of pull from _____ attachments to the _____ part of the globe (eyeball)"
back = "posterior, anterior"

def reduce_dimensions(target_dim):
    # load embeddings from JSON file
    with open("embeddings.json", 'r') as f:
        embeddings_dict = json.load(f)

    # keys + embeddings
    text_list = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[t] for t in text_list])

    # reduce dimensionality
    pca = PCA(n_components=target_dim)
    reduced = pca.fit_transform(embeddings)

    # new dict with dim-reduced embeddings
    reduced_dict = {text_list[i]: list(reduced[i]) for i in range(len(text_list))}

    # Save to JSON file
    with open(f'embeddings_{target_dim}.json', 'w') as f:
        json.dump(reduced_dict, f)


reduce_dimensions(10)