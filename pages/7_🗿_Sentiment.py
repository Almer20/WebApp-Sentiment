import csv
import pymysql
import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
import joblib
import plotly.express as px  # Import Plotly for advanced charting

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to connect to MySQL database
def connect_to_db():
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='',  # Isi dengan password MySQL Anda
        database='webanalysisdemo',
        charset='utf8mb4',  # Menentukan charset untuk mendukung emoji, dll.
        cursorclass=pymysql.cursors.DictCursor  # Menggunakan DictCursor untuk mendapatkan data dalam bentuk kamus
    )
    return conn

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove numbers and special characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Convert to lowercase and split into tokens
    tokens = nltk.word_tokenize(text.lower())
    
    # Remove punctuation and stopwords, and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    
    # Join tokens back into string
    text = ' '.join(tokens)
    
    return text

# Function to remove stopwords
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

# Function to perform stemming
def stem_text(text):
    porter = PorterStemmer()
    stemmed_words = [porter.stem(word) for word in word_tokenize(text)]
    return ' '.join(stemmed_words)

# Function to insert data into MySQL database
def insert_data(conn, data):
    cursor = conn.cursor()
    
    # Mengambil nomor urut terakhir dari database
    last_review_id = 0
    try:
        cursor.execute("SELECT MAX(Review_id) FROM reviews")
        result = cursor.fetchone()
        if result['MAX(Review_id)']:
            last_review_id = result['MAX(Review_id)']
    except pymysql.Error as e:
        st.error(f"Error fetching last review ID: {e}")
        return
    
    # Menambahkan data dengan nomor urut yang tepat
    for row in data:
        row['Review_id'] = last_review_id + 1
        last_review_id += 1
    
    insert_query = """
    INSERT INTO reviews 
    (Review_id, Title, DatePosted, ReviewText)
    VALUES (%(Review_id)s, %(Title)s, %(DatePosted)s, %(ReviewText)s)
    """
    try:
        cursor.executemany(insert_query, data)
        conn.commit()
    except pymysql.Error as e:
        st.error(f"Error inserting data: {e}")

# Function to delete all data from MySQL database
def delete_all_data(conn):
    cursor = conn.cursor()
    delete_query = """
    DELETE FROM reviews
    """
    try:
        cursor.execute(delete_query)
        conn.commit()
    except pymysql.Error as e:
        st.error(f"Error deleting data: {e}")

def main():
    st.title("Upload Review Pengguna Steam ")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read CSV file
        try:
            csv_data = csv.DictReader(uploaded_file.read().decode('utf-8').splitlines(), delimiter=';')
            df = pd.DataFrame(csv_data)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return

        # Konversi format tanggal
        df['DatePosted'] = pd.to_datetime(df['DatePosted'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')

        # Display table
        df['sentiment'] = ''
        for index, row in df.iterrows():        
            ReviewText = row['ReviewText']
            cleaned_text = preprocess_text(ReviewText)
            tokens_casefolding = cleaned_text.split() # Tokenization and Case Folding
            stopword_text = remove_stopwords(tokens_casefolding) # Remove stopwords
            stemming_text = stem_text(cleaned_text) # Stemming
            transformed_text = vectorizer.transform([stemming_text])
            predictions = model.predict(transformed_text)
            df.loc[index, 'sentiment'] = predictions[0]  # Ensure it's a single value, not a list

        # Show the dataframe
        st.write(df)

        # Plot sentiment distribution
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig = px.bar(sentiment_counts, x='Sentiment', y='Count', labels={'Sentiment': 'Sentiment', 'Count': 'Count'}, title='Sentiment Distribution')
        st.plotly_chart(fig)

        # Connect to MySQL database
        conn = connect_to_db()

        if conn:
            # Insert data into MySQL database
            # insert_data(conn, df.to_dict('records'))
            st.success("Data successfully uploaded to MySQL database.")
    
if os.path.exists('sentiment_model.pkl') and os.path.exists('vectorizer.pkl'):
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
else:
    model = None
    vectorizer = None
    st.error("Model Belum ada lakukan pelatihan dulu")

if __name__ == "__main__":
    main()
