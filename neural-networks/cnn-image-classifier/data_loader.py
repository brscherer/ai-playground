import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("data/IMDB Dataset.csv")

    df["sentiment"] = df["sentiment"].map({
        "positive": 1,
        "negative": 0
    })

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["review"].tolist(),
        df["sentiment"].tolist(),
        test_size=0.2
    )

    return train_texts, val_texts, train_labels, val_labels