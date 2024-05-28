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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

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


def main():
    st.title("Klasifikasi Naive Bayes 🎃 ")

    # Connect to MySQL database
    conn = connect_to_db()
    
    if conn:
        cursor = conn.cursor()

        # Perform classification
        if st.button("Classify Data"):
            cursor.execute("SELECT stemming_data, sentiment FROM labelling")
            classification_data = cursor.fetchall()
            df_cleaned = pd.DataFrame(classification_data)
            
            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(df_cleaned['stemming_data'], df_cleaned['sentiment'], test_size=0.3, random_state=42)
            
            # Vectorize the text data
            vectorizer = CountVectorizer()
            X_train_vectorized = vectorizer.fit_transform(X_train)
            X_test_vectorized = vectorizer.transform(X_test)
            
            st.subheader('Hasil Ekstraksi Fitur')
            st.write('-------------------')
            st.write('Vektor fitur Data Latih:')
            st.write(pd.DataFrame(X_train_vectorized.toarray(), columns=vectorizer.get_feature_names_out()))
            st.write('-------------------')
            st.write('\nVektor fitur Data Uji:')
            st.write(pd.DataFrame(X_test_vectorized.toarray(), columns=vectorizer.get_feature_names_out()))
            
            # Train the model
            model = MultinomialNB()
            model.fit(X_train_vectorized, y_train)
            
            # Make predictions
            predictions = model.predict(X_test_vectorized)

            # Save the model and vectorizer
            joblib.dump(model, 'sentiment_model.pkl')
            joblib.dump(vectorizer, 'vectorizer.pkl')
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, predictions)
            st.write(f'Accuracy: {accuracy:.2f}')
            
            # Display classification report
            st.subheader('Classification Report')
            st.write(classification_report(y_test, predictions))
            
            # Display confusion matrix
            st.subheader('Confusion Matrix')
            conf_matrix = confusion_matrix(y_test, predictions)
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Netral', 'Positif'], yticklabels=['Negatif', 'Netral', 'Positif'], ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Prediksi')
            ax.set_ylabel('Aktual')
            st.pyplot(fig)

if __name__ == "__main__":
    main()
