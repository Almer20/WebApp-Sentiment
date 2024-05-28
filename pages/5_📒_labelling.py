import streamlit as st
import pandas as pd
import pymysql
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download resources for NLTK
nltk.download('punkt')
nltk.download('vader_lexicon')

# Function to connect to MySQL database
def connect_to_db():
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='',  # Fill in your MySQL password
        database='webanalysisdemo',
        charset='utf8mb4',  # Specify charset to support emojis, etc.
        cursorclass=pymysql.cursors.DictCursor  # Use DictCursor to get data as dictionaries
    )
    return conn

# Function for sentiment analysis and labeling
def perform_sentiment_analysis(text):
    data = SentimentIntensityAnalyzer()
    sentiment_scores = data.polarity_scores(text)
    return sentiment_scores['compound']

def main():
    st.title("Sentiment Analysis and Labeling")

    # Connect to MySQL database
    conn = connect_to_db()
    
    if conn:
        cursor = conn.cursor()

        # Button to start labelling process
        if st.button("Mulai Labelling"):
            # Retrieve data from preprocessing table
            cursor.execute("SELECT Review_id, Title, DatePosted, stemming_data FROM preprocessing")
            data = cursor.fetchall()
            df = pd.DataFrame(data)

            scores = []
            labels = []
            for index, row in df.iterrows():
                score = perform_sentiment_analysis(row['stemming_data'])
                scores.append(score)
                
                if score > 0:
                    label = 'positif'
                elif score < 0:
                    label = 'negatif'
                else:
                    label = 'netral'
                
                labels.append(label)

            # Add sentiment scores and labels to the dataframe
            df['sentiment_score'] = scores
            df['sentiment'] = labels

            # Save labeled data to labelling table in the database
            try:
                with conn.cursor() as cursor:
                    # Drop table if exists
                    cursor.execute("DROP TABLE IF EXISTS labelling")

                    # Create labelling table
                    create_table_query = """
                    CREATE TABLE labelling (
                        Review_id INT AUTO_INCREMENT PRIMARY KEY,
                        Title VARCHAR (255),
                        DatePosted DATE,
                        stemming_data TEXT,
                        sentiment_score FLOAT,
                        sentiment VARCHAR(10)
                    )
                    """
                    cursor.execute(create_table_query)

                    # Insert labeled data into labelling table
                    for index, row in df.iterrows():
                        insert_query = "INSERT INTO labelling (Review_id, Title, DatePosted, stemming_data, sentiment_score, sentiment) VALUES (%s,%s,%s, %s, %s, %s)"
                        cursor.execute(insert_query, (row['Review_id'], row['Title'], row['DatePosted'], row['stemming_data'], row['sentiment_score'], row['sentiment']))

                # Commit changes
                conn.commit()
                st.success("Labeled data saved to database.")
            except Exception as e:
                st.error(f"Error: {e}")

        # Retrieve data from preprocessing table
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM preprocessing")
        preprocessing_data = cursor.fetchall()
        preprocessing_df = pd.DataFrame(preprocessing_data)

        # Display preprocessing data
        st.subheader("Preprocessing Data")
        st.write(preprocessing_df)

        # Retrieve data from labelling table
        cursor.execute("SELECT * FROM labelling")
        labelling_data = cursor.fetchall()
        labelling_df = pd.DataFrame(labelling_data)

        # Display labeled data
        st.subheader("Labeled Data")
        st.write(labelling_df)

        # Button to delete data from labelling table
        if st.button("Delete Data from Labelling Table"):
            try:
                with conn.cursor() as cursor:
                    # Delete all rows from labelling table
                    cursor.execute("DELETE FROM labelling")
                    conn.commit()
                    st.success("Data deleted from labelling table.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
