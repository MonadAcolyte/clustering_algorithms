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

# Import the preprocessing modules
from preprocessing import load_data, clean_data

# Working env. defining & importing and cleaning the data
filepath = sys.argv[1]
raw = load_data(filepath)
cleaned = clean_data(raw)
vectorised_sentences = cleaned["Sentence"].tolist()

# Vectorisation
def vectorise(sentences: list) -> tuple:
    """Convert cleaned sentences to a TF-IDF matrix (tuple py variable type)."""
    vectorizer = TfidfVectorizer(
        stop_words="english",  # This will get rid of structural words I.E. Articles, common pronouns, &c.
        max_features=4500,     # Keep only the 4500 most important terms, 4500 was picked via trial and error...
        min_df=2,              # Ignores words appearing in fewer than 2 sentences
    )
    matrix = vectorizer.fit_transform(sentences)
    return matrix, vectorizer

matrix, vectorizer = vectorise(vectorised_sentences)

# Silhouette analysis (w/ K means)
def silhouette_analysis(matrix, k_range: range) -> dict:
    """Compute silhouette scores for each K and plot the results."""
    scores = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix)
        score = silhouette_score(matrix, labels)
        scores[k] = score
        print(f"K={k}: silhouette score = {score:.4f}")

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(list(scores.keys()), list(scores.values()), marker="o", color = "darkmagenta")
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