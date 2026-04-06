# NOTE: This crashes if run alone, remember to use the "run" shell scripts that match the pdf instructions

# Libraries
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD # (LSA implementation)

# Import the preprocessing modules
from preprocessing import load_data, clean_data

###################
# Basic Functions #
###################

# Vectorisation
def vectorise(sentences: list) -> tuple:
    """Convert cleaned sentences to a TF-IDF matrix (tuple py variable type)."""
    vectoriser = TfidfVectorizer(
        stop_words="english",  # This will get rid of structural words I.E. Articles, common pronouns, &c.
        max_features=4500,     # Keep only the 4500 most important terms, 4500 was picked via trial and error...
        min_df=2,              # Ignores words appearing in fewer than 2 sentences
    )
    matrix = vectoriser.fit_transform(sentences)
    return matrix, vectoriser

# LSA
def apply_lsa(matrix, n_components: int = 100) -> np.ndarray:
    """Reduce TF-IDF matrix dimensions using LSA."""
    lsa = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_matrix = lsa.fit_transform(matrix)
    explained_variance = lsa.explained_variance_ratio_.sum()
    print(f"LSA: {n_components} components explain {explained_variance:.2%} of variance")
    return reduced_matrix

############
# Pipeline #
############

# Working env. defining & importing and cleaning the data
filepath = sys.argv[1]
raw = load_data(filepath)
cleaned = clean_data(raw)
vectorised_sentences = cleaned["Sentence"].tolist()

# Apply Vectorisation
matrix, vectoriser = vectorise(vectorised_sentences)

# Apply LSA
matrix = apply_lsa(matrix)

# Silhouette analysis (w/ K means)
def silhouette_analysis(matrix, k_range: range) -> dict:
    """Compute silhouette scores for each K and plot the results."""
    scores = {}
    scores[1] = 0 # Obv. for k=1 the s. coeff.'s value is 0
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix)
        score = silhouette_score(matrix, labels)
        scores[k] = score
        print(f"K={k}: silhouette score = {score:.4f}")
    
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(list(scores.keys()), list(scores.values()), marker="o", color = "yellowgreen")
    plt.grid(True, alpha = 0.5)
    plt.xlabel("K")
    plt.ylabel("Silhouette Coefficient")
    plt.title("Silhouette Analysis")
    plt.xticks(list(scores.keys()))
    plt.tight_layout()
    plt.savefig("silhouette_plot.png")
    print("Saved silhouette_plot.png")

    return scores

scores = silhouette_analysis(matrix, k_range=range(2, 11))

# Optimal K clustering
optimal_k = max(scores, key=scores.get)
print(f"\nOptimal K: {optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
final_labels = kmeans.fit_predict(matrix)

# Label text file
with open("label.txt", "w") as f:
    for label in final_labels:
        f.write(f"{label + 1}\n")  # +1 so labels start at 1, not 0

print("Saved label.txt")