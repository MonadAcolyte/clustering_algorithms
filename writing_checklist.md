# Writing checklist

## Preprocessing Report

- [] Brief description of the dataset (1,760 sentences, mixed sources)
- [] Steps applied: special character/punctuation removal, \n literal removal, whitespace normalisation, lowercasing
- [] Why each step was applied
- [] Mention of min_df=2 and max_features=4500 in TF-IDF and why
- [] How these steps may influence clustering (e.g. lowercasing prevents "The" and "the" being treated as different terms)
- [] Note on the 5 empty rows and how they were handled

## Clustering Report

- [] Algorithm used: K-Means, brief explanation of how it works
- [] Vectorisation: TF-IDF, brief explanation
- [] Dimensionality reduction: LSA via TruncatedSVD, why it was needed
- [] The n_components trade-off (you can use that table here)
- [] Silhouette analysis explanation — what it measures, why K=1 is hardcoded to 0
- [] Chosen optimal K and justification
- [] Cluster inspection findings — mention the semantic coherence observed
- [] Honest acknowledgement of why scores are low overall (mixed dataset, curse of dimensionality)