import streamlit as st
import pandas as pd
import pymysql
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download stopwords jika belum diunduh sebelumnya
nltk.download('stopwords')
nltk.download('punkt')
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

# Function to preprocess text
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

# Function to preprocess user reviews and store in a new table
def preprocess_and_store(conn):
    cursor = conn.cursor()
    select_query = "SELECT Review_id, Title, DatePosted, ReviewText FROM reviews"
    cursor.execute(select_query)
    reviews = cursor.fetchall()
    
    for review in reviews:
        Review_id = review['Review_id']
        
        # Check if Review_id already exists in preprocessing table
        cursor.execute("SELECT 1 FROM preprocessing WHERE Review_id = %s", (Review_id,))
        if cursor.fetchone() is None:
            Title = review['Title']
            DatePosted = review['DatePosted']
            ReviewText = review['ReviewText']
            cleaned_text = preprocess_text(ReviewText)
            tokens_casefolding = cleaned_text.split() # Tokenization and Case Folding
            stopword_text = remove_stopwords(tokens_casefolding) # Remove stopwords
            stemming_text = stem_text(cleaned_text) # Stemming
            insert_query = """INSERT INTO preprocessing 
                            (Review_id, Title, DatePosted, ReviewText, cleansing, token_casefolding, stemming_data, stopword) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(insert_query, (Review_id, Title, DatePosted, ReviewText, cleaned_text, ' '.join(tokens_casefolding), stemming_text, ' '.join(stopword_text)))
    
    conn.commit()

# Function to clear data in the preprocessing table
def clear_preprocessing_data(conn):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM preprocessing")
    conn.commit()

def main():
    st.title("Preprocessing")

    # Connect to MySQL database
    conn = connect_to_db()
    
    if conn:
        cursor = conn.cursor()
        
        # Check if preprocessing table exists
        cursor.execute("SHOW TABLES LIKE 'preprocessing'")
        table_exists = cursor.fetchone()
        
        if table_exists:
            # Display preprocessing table
            st.subheader("Preprocessing Reviews Table")
            cursor.execute("SELECT * FROM preprocessing")
            preprocessing_data = cursor.fetchall()
            preprocessing_df = pd.DataFrame(preprocessing_data)
            st.write(preprocessing_df)

            
            # Button to clear preprocessing table data
            if st.button("Clear Preprocessing Table Data"):
                clear_preprocessing_data(conn)
                st.success("All data in the preprocessing table has been cleared.")
                st.experimental_rerun()
        
        else:
            st.write("Preprocessing table does not exist.")
            # Create preprocessing table if it doesn't exist
            cursor.execute("""CREATE TABLE preprocessing (
                                Review_id INT PRIMARY KEY,
                                Title VARCHAR(255),
                                DatePosted DATE,
                                ReviewText TEXT,
                                cleansing TEXT,
                                token_casefolding TEXT,
                                stopword TEXT,
                                stemming_data TEXT
                            )""")
            conn.commit()
            st.success("Preprocessing table created.")

        # Button to preprocess and store data
        if st.button("Preprocess Data"):
            preprocess_and_store(conn)
            st.success("Data preprocessing complete.")
            st.experimental_rerun()
            
            # Display new table with preprocessed data
            st.subheader("Updated Preprocessed Reviews Table")
            cursor.execute("SELECT * FROM preprocessing")
            updated_preprocessing_data = cursor.fetchall()
            updated_preprocessing_df = pd.DataFrame(updated_preprocessing_data)
            st.write(updated_preprocessing_df)


if __name__ == "__main__":
    main()
    