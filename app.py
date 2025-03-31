import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import joblib


# Load dataset (Ensure it has 'text' and 'label' columns)
df = pd.read_csv("IMDB Dataset.csv", escapechar='\\') 
model = joblib.load('senti_analysis.pkl')
# Convert labels to numeric (0 = Negative, 1 = Neutral, 2 = Positive)
label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
df["sentiment"] = df["sentiment"].map(label_mapping)

# Split dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(df["review"], df["sentiment"], test_size=0.2, random_state=42)

MAX_WORDS = 10000  # Vocabulary size
MAX_LEN = 100  # Max sequence length

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding="post")
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN, padding="post")


import requests
from bs4 import BeautifulSoup

def fetch_rotten_tomatoes_reviews(movie_url, limit=10):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(movie_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    reviews = []
    review_elements = soup.find_all("p", class_="review-text") # Adjust class as needed

    for review in review_elements[:limit]:
        reviews.append(review.get_text(strip=True))

    return reviews

# Example Rotten Tomatoes URL (Replace with actual movie URL)
movie_url = "https://www.rottentomatoes.com/m/london_fields/reviews"
reviews = fetch_rotten_tomatoes_reviews(movie_url, limit=10)
reviews = [review for review in reviews if review]
print(reviews)  # Check the scraped reviews




