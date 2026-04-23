import sys
import re
import csv
import nltk
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap

# Configurations, hyperparams, &c.

DEFAULT_DATA_PATH = "data.txt"
SBERT_MODEL       = "all-MiniLM-L6-v2" # I picked this one because it's light enough to run on my machine
UMAP_COMPONENTS   = 10
UMAP_NEIGHBOURS   = 15
K_RANGE           = range(2, 11)
KMEANS_INIT       = 25
RANDOM_STATE      = 2112

# Load data

filepath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH

df = pd.read_csv(
    filepath,
    sep="\t",
    names=["ID", "Sentence"],
    skiprows=1,
    encoding="utf-8",
    engine="python",
    quoting=csv.QUOTE_NONE,
)

# Preprocessing with spaCy and NLTK

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
stop_words = set(stopwords.words("english"))

def preprocess(text):
    # Remove literal \n sequences and normalise whitespace
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text).lower().strip()
    # Lemmatise with spaCy, filter stopwords and short tokens with NLTK stopword list
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if token.lemma_ not in stop_words and len(token.lemma_) > 2
    ]
    return " ".join(tokens) if tokens else "empty"

print("Preprocessing... please wait")
df["Cleaned"] = df["Sentence"].astype(str).apply(preprocess)
sentences = df["Cleaned"].tolist()

# SBERT embeddings

print("SBERT embeddings are being generated... ... ... this may take an aeon or two")
model      = SentenceTransformer(SBERT_MODEL)
embeddings = model.encode(sentences, show_progress_bar=True)

# UMAP dimensionality reduction

print("Applying UMAP...")
reducer   = umap.UMAP(
    n_components = UMAP_COMPONENTS,
    n_neighbors  = UMAP_NEIGHBOURS,
    random_state = RANDOM_STATE,
)
reduced = reducer.fit_transform(embeddings)

# Silhouette analysis

print("Running silhouette analysis... \n \n")
scores    = {1: 0}  # K=1 is undefined for silhouette, hardcoded as 0
for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=KMEANS_INIT)
    labels = kmeans.fit_predict(reduced)
    score  = silhouette_score(reduced, labels)
    scores[k] = score
    print(f"K={k}: silhouette score = {score:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(list(scores.keys()), list(scores.values()), marker="o", color="steelblue")
plt.grid(True, alpha=0.5)
plt.xlabel("K")
plt.ylabel("Silhouette Coefficient")
plt.title("Silhouette Analysis")
plt.xticks(list(scores.keys()))
plt.tight_layout()
plt.savefig("silhouette_plot.png")
print("Saved plot as: silhouette_plot.png in the current directory")

# Cluster with optimal K

optimal_k = max(scores, key=scores.get)
print(f"\nOptimal K: {optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=KMEANS_INIT)
final_labels = kmeans.fit_predict(reduced)

# Output label.txt

with open("label.txt", "w") as f:
    for label in final_labels:
        f.write(f"{label + 1}\n")

print("Saved label.txt, in the current directory")

# Inspection snippet; I.E. *of the manual kind*

print("\n ¡ Cluster Inspection ! \n")
for cluster_id in range(1, optimal_k + 1):
    indices = [i for i, label in enumerate(final_labels) if label + 1 == cluster_id]
    print(f"\nCluster {cluster_id} ({len(indices)} sentences):")
    for i in indices[:10]: # To show more sentences, increase * only * this number!
        print(f"  - {sentences[i]}")