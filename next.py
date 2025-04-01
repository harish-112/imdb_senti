import psycopg2
from keras.models import load_model
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("senti_analysis.h5")

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

def analyze_sentiment(text):
    seq = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_LEN, padding="post")
    prediction = model.predict(seq)
    print(prediction)
    sentiment_label = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_label[np.argmax(prediction)]

# Predict sentiment for Rotten Tomatoes reviews
for review in reviews:
    print(f"Review: {review}\nSentiment: {analyze_sentiment(review)}\n")

DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "interlace"
DB_HOST = "localhost"  
DB_PORT = "5432"

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cursor = conn.cursor()

# Create table for storing movie reviews and sentiment
cursor.execute("""
    CREATE TABLE IF NOT EXISTS movie_reviews (
        id SERIAL PRIMARY KEY,
        review TEXT,
        sentiment VARCHAR(10)
    )
""")
conn.commit()

def store_reviews_in_db(reviews):
    for review in reviews:
        sentiment = analyze_sentiment(review)
        cursor.execute("INSERT INTO movie_reviews (review, sentiment) VALUES (%s, %s)", (review, sentiment))
    
    conn.commit()
    print("Reviews and sentiments saved successfully!")

# Store Rotten Tomatoes reviews in PostgreSQL
store_reviews_in_db(reviews)
