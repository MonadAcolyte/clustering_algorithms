# The main script uses this function

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