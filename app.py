import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load dataset (Ensure it has 'text' and 'label' columns)
df = pd.read_csv("/content/IMDB Dataset.csv", escapechar='\\') 

# Convert labels to numeric (0 = Negative, 1 = Neutral, 2 = Positive)
label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
df["sentiment"] = df["sentiment"].map(label_mapping)

# Split dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(df["review"], df["sentiment"], test_size=0.2, random_state=42)
