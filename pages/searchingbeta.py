import streamlit as st
import pandas as pd
import pymysql
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function to connect to MySQL database
def connect_to_db():
    try:
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='',
            database='webanalysisdemo',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return conn
    except pymysql.Error as e:
        st.error(f"Error connecting to database: {e}")
        return None

# Function to fetch data from MySQL database
def fetch_data_from_mysql(conn):
    cursor = conn.cursor()
    select_query = "SELECT Review_id, Title, DatePosted, stemming_data, sentiment_score, sentiment FROM labelling;"
    try:
        cursor.execute(select_query)
        data = cursor.fetchall()
        if not data:
            st.warning("No data retrieved from the database.")
        return data
    except pymysql.Error as e:
        st.error(f"Error fetching data: {e}")
        return None

def display_sentiment_count(df):
    """
    Function to display sentiment counts.
    """
    sentiment_counts = df['prediction'].value_counts()
    st.write("Sentiment Counts:")
    st.write(sentiment_counts)

def sentiment_analysis(conn, cursor, actual_sentiments, data):
    if not data:
        st.error("No data available for sentiment analysis.")
        return

    df = pd.DataFrame(data)

    required_columns = ['Review_id', 'Title', 'DatePosted', 'stemming_data', 'sentiment']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Missing column in the data: {col}")
            return

    # Add the actual sentiment labels to the DataFrame
    df['actual'] = actual_sentiments[:len(df)]  # Slice actual_sentiments to match the length of df

    corpus = df['stemming_data'].tolist()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    # Assuming we have sentiment labels in the dataset
    y = df['sentiment']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.write(f'Accuracy: {accuracy}')
    st.write('Classification Report:\n', report)

    # Display confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Actual Labels')
    st.pyplot(fig)
    
    # Store results back to database
    df['prediction'] = model.predict(X)  # Adding 'prediction' column to DataFrame
    
    insert_query = """
    INSERT INTO class (Review_id, Title, DatePosted, stemming_data, actual, prediction)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    
    for i, row in df.iterrows():
        cursor.execute(insert_query, (
            row['Review_id'],
            row['Title'],
            row['DatePosted'],
            row['stemming_data'],
            row['actual'],  
            row['prediction']
        ))
    
    conn.commit()
    st.success("Sentiment analysis completed and data inserted successfully.")

    display_sentiment_count(df)
    st.write("Processed Data:")
    st.dataframe(df)


def delete_data_from_class(conn):
    cursor = conn.cursor()
    delete_query = "DELETE FROM class"
    try:
        cursor.execute(delete_query)
        conn.commit()
        st.success("All data deleted from the class table.")
    except pymysql.Error as e:
        st.error(f"Error deleting data: {e}")

def check_duplicate_data(conn, search_data):
    cursor = conn.cursor()
    select_query = "SELECT Review_id, Title, DatePosted, stemming_data, actual, prediction FROM class WHERE Review_id = %s;"
    for row in search_data:
        cursor.execute(select_query, (row['Review_id'],))
        existing_data = cursor.fetchone()
        if existing_data:
            st.warning(f"Duplicate data found for Review ID: {row['Review_id']}")
            st.write("Existing Data:")
            st.write(existing_data)

def main():
    st.title("Sentiment Analysis Web App")

    conn = connect_to_db()

    if conn:
        # st.write("Fetching data from the database...")
        data = fetch_data_from_mysql(conn)

        if data:
            # Fetch actual sentiments from the database
            actual_sentiments = [row['sentiment'] for row in data]

            # Search and classify based on user input
            keyword = st.text_input("Enter keyword to search and classify:")
            if st.button('Search and Classify'):
                st.write(f"Searching and classifying data for keyword: {keyword}")
                cursor = conn.cursor(pymysql.cursors.DictCursor)
                search_query = f"SELECT Review_id, Title, DatePosted, stemming_data, sentiment FROM labelling WHERE Title LIKE '%{keyword}%';"
                cursor.execute(search_query)
                search_data = cursor.fetchall()
                check_duplicate_data(conn, search_data)
                sentiment_analysis(conn, cursor, actual_sentiments, search_data)

            if st.button('Delete all data from class table'):
                delete_data_from_class(conn)    

            conn.close()
        else:
            st.error("No data found in the database.")
    else:
        st.error("Failed to connect to the database.")

def check_duplicate_data(conn, search_data):
    cursor = conn.cursor()
    select_query = "SELECT Review_id, Title, DatePosted, stemming_data, actual, prediction FROM class WHERE Review_id = %s;"
    for row in search_data:
        cursor.execute(select_query, (row['Review_id'],))
        existing_data = cursor.fetchone()
        if existing_data:
            st.warning(f"Duplicate data found for Review ID: {row['Review_id']}")
            st.write("Existing Data:")
            st.write(existing_data)
            return True
    return False

if __name__ == "__main__":
    main()
