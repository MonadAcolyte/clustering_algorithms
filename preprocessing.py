import re
import csv
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load the tab-separated data file into a DataFrame."""
    df = pd.read_csv(
        filepath,
        sep="\t",
        names=["ID", "Sentence"],
        skiprows=1, # skip the header row
        encoding="utf-8",
        engine="python",
        quoting=csv.QUOTE_NONE # This makes Pandas ignore the double quotes in the text
    )
    return df


def clean_sentence(text: str) -> str:
    """Clean a single sentence."""
    # Remove literal \n escape sequences (e.g. \\n\\n in row 11)
    text = text.replace("\\n", " ")
    # Remove real newline/carriage return characters (\r\n Windows line endings)
    text = text.replace("\n", " ").replace("\r", " ")
    # Remove punctuation and special characters, keeping only letters, digits, spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Collapse multiple spaces into one and strip leading/trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Lowercase
    text = text.lower()
    return text


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning to the Sentence column."""
    df = df.copy()
    df["Sentence"] = df["Sentence"].astype(str).apply(clean_sentence)
    # Drop rows where cleaning left an empty sentence
    df = df[df["Sentence"].str.len() > 0].reset_index(drop=True)
    return df


if __name__ == "__main__":
    raw = load_data("data/data_train.txt")
    print(f"Loaded {len(raw)} rows")
    print(raw.head())

    cleaned = clean_data(raw)
    print(f"\nCleaned: {len(cleaned)} rows remaining")
    print(cleaned.head())

    cleaned.to_csv("data/data_train_cleaned.csv", index=False)
    print("\nSaved to data/data_train_cleaned.csv")