import streamlit as st
import pandas as pd
import pymysql
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter


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
        
# Function to fetch data from MySQL database
def fetch_data_from_mysql(conn):
    cursor = conn.cursor()
    select_query = """
    SELECT * FROM reviews
    """
    try:
        cursor.execute(select_query)
        data = cursor.fetchall()
        return data
    except pymysql.Error as e:
        st.error(f"Error fetching data: {e}")

def main():
    st.title("Reviews Pengguna Steam")

    # Tombol untuk menghapus data dari database
    if st.button("Hapus Semua Data Review"):
        conn = connect_to_db()
        if conn:
            delete_all_data(conn)
            st.success("Semua data berhasil dihapus dari database.")

    # Tampilkan data dari MySQL database
    conn = connect_to_db()
    if conn:
        data = fetch_data_from_mysql(conn)
        if data:
            df_data = pd.DataFrame(data)
            st.write("Data from MySQL Database:")
            with st.expander("Data Preview"):
                st.write(df_data)
        else:
            st.info("Tidak ada data yang tersedia.")

        # Retrieve data from preprocessing table
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM preprocessing")
        preprocessing_data = cursor.fetchall()
        preprocessing_df = pd.DataFrame(preprocessing_data)

        # Display preprocessing data
        st.subheader("Preprocessing Data")
        with st.expander("Data Preview"):
            st.write(preprocessing_df)
        

          # Retrieve data from labelling table
        cursor.execute("SELECT * FROM labelling")
        labelling_data = cursor.fetchall()
        labelling_df = pd.DataFrame(labelling_data)

        # Display labeled data
        st.subheader("Labeled Data")
        with st.expander("Data Preview"):
            st.write(labelling_df)
            
        # Hitung jumlah setiap sentimen pada stemming data
        # sentiment_counts = labelling_df['sentiment'].value_counts()

        # Tampilkan jumlah setiap sentimen pada stemming data
        st.subheader("Jumlah Setiap Sentimen pada Stemming Data")
        # st.write(sentiment_counts)

        # Hitung persentase masing-masing sentimen
        total_count = len(labelling_df)
        # percentages = (sentiment_counts / total_count * 100).round(2)

        # Tampilkan persentase masing-masing sentimen
        st.subheader("Persentase Setiap Sentimen pada Stemming Data")
        # st.write(percentages)

        # Tampilkan total jumlah keseluruhan data dalam stemming data
        st.subheader("Total Jumlah Keseluruhan Data dalam Stemming Data")
        st.write(total_count)

        # Generating WordCloud and Plot Frequency
        if st.button("Generate WordCloud and Plot Frequency"):
            df_cleaned = labelling_df.copy()
            df_cleaned['stemming_data'] = df_cleaned['stemming_data'].astype(str)

            df_text = ''.join(df_cleaned['stemming_data'].tolist())
            stopwords = set(STOPWORDS)
            stopwords.update(['https', 'the', 'of', 'to', '...', 'amp'])
            wc = WordCloud(stopwords=stopwords, background_color="black", max_words=500, width=800, height=400)
            wc.generate(df_text)

            tokens = df_text.split()
            word_counts = Counter(tokens)
            top_words = word_counts.most_common(10)
            word,count = zip(*top_words)

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            axes[0].imshow(wc, interpolation='bilinear')
            axes[0].axis("off")
            axes[0].set_title("WordCloud")

            colors = plt.cm.Paired(range(len(word)))
            bars = axes[1].bar(word, count, color=colors)
            axes[1].set_xlabel("kata")
            axes[1].set_ylabel("frekuensi")
            axes[1].set_title("Frekuensi Kata-kata")
            axes[1].tick_params(axis='x', rotation=45)

            for bar, num in zip(bars, count):
                axes[1].text(bar.get_x() + bar.get_width() / 2 - 0.1, num + 1, str(num), fontsize=10, color='black', ha='center')

            plt.tight_layout()

            # Displaying plot using streamlit
            st.pyplot(fig)



if __name__ == "__main__":
    main()
