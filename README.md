# clustering_algorithms

This repository attempts to cluster a sentence data set.

The available methods are:
    - TF-IDF with KMEANS
    - SBERT with UMAP

Read below for an updated debrief. LaTex reports are available on each, but are outdated.

# SBERT (Sentence-BERT)

Pretrained encoder with masked language modelling. For every sentence (as a whole) it produces a context vector, as well as additional vectors for every sentence. When comparing two sentences it will consider the all of these context vectors, i.e. they will all be passed through the network. SBERT also incorporates a pooling layer that pools from the vectors and finetunes the network over a large selection of sentence-pairs.

Note on the SBERT report: it confused n components and n neighbours. For clarity, it's `N_NEIGHBOURS` that controls the balance between local and global structure.

## Preprocessing:
SBERT doesn't require Lemmatising and removal of stopwords, it is already trained on natural language.

`nltk.download("stopwords")` needs to be run once for `stopwords` to work.

# K-means

Note on the K-means report: The actual algorithm takes a single k value, and I then plot the outcome over a selected array of values. It also minimises the sum of the *squared* distance to centroid values, aka inertia (I forgot squared on the report). The sklearn default for amount of initialisations is 25, it thus scatters the points that many times and pick the scenario with the lowest inertia. Also for clarity, *before* tuning `UMAP`, the academic sentences would be so far apart that everything else would cluster together. Raising n neighbours fixed that.

