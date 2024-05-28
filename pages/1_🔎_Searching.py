import streamlit as st
import pandas as pd
import pymysql
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from st_pages import hide_pages

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

# Function to add custom CSS
def add_css():
    st.markdown(
        """
        <style>
        .dataframe-container {
            width: 1000px;
            overflow-x: auto;
        }
        .dataframe table {
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    st.title("Sentiment Analysis Game")
    
    # Add custom CSS
    add_css()

    # Connect to MySQL database
    conn = connect_to_db()
    
    if conn:
        cursor = conn.cursor()

        # Retrieve data from labelling table
        cursor.execute("SELECT * FROM labelling")
        labelling_data = cursor.fetchall()
        
        if len(labelling_data) == 0:
            st.write("Tidak ada data yang tersedia pada tabel labelling.")
        else:
            labelling_df = pd.DataFrame(labelling_data)

            # Display labeled data
            # st.subheader("Labeled Data")
            # st.write(labelling_df)

            # Search for sentiment by game title
            search_term = st.text_input("Search game title:")
            if search_term:
                game_df = labelling_df[labelling_df['Title'].str.contains(search_term, case=False)]
                if not game_df.empty:
                    total_count = len(game_df)
                    st.subheader(f"Sentiment Analysis for '{search_term}' ({total_count} records found)")
                    
                    # Adjust table width using CSS
                    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                    st.dataframe(game_df)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Calculate sentiment count
                    sentiment_count = game_df['sentiment'].value_counts()

                    # Display sentiment count
                    st.write("Sentiment counts:")
                    st.write(sentiment_count)

                    # Plot bar chart for sentiment count
                    sns.set_style('whitegrid')
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax = sns.barplot(x=sentiment_count.index, y=sentiment_count.values, hue=sentiment_count.index, palette='pastel', legend=False)
                    plt.title('Jumlah Analisis Sentimen', fontsize=14, pad=20)
                    plt.xlabel('Class Sentiment', fontsize=12)
                    plt.ylabel('Jumlah Review', fontsize=12)

                    for i, count in enumerate(sentiment_count.values):
                        ax.text(i, count + 0.10, str(count), ha='center', va='bottom')

                    st.pyplot(fig)

                    # Generate WordCloud and frequency bar chart
                    game_df['stemming_data'] = game_df['stemming_data'].astype(str)
                    df_text = ''.join(game_df['stemming_data'].tolist())
                    stopwords = set(STOPWORDS)
                    stopwords.update(['https', 'the', 'of', 'to', '...', 'amp'])
                    wc = WordCloud(stopwords=stopwords, background_color="black", max_words=500, width=800, height=400)
                    wc.generate(df_text)

                    tokens = df_text.split()
                    word_counts = Counter(tokens)
                    top_words = word_counts.most_common(10)
                    word, count = zip(*top_words)

                    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                    axes[0].imshow(wc, interpolation='bilinear')
                    axes[0].axis("off")
                    axes[0].set_title("WordCloud")

                    colors = plt.cm.Paired(range(len(word)))
                    bars = axes[1].bar(word, count, color=colors)
                    axes[1].set_xlabel("Kata")
                    axes[1].set_ylabel("Frekuensi")
                    axes[1].set_title("Frekuensi Kata-kata")
                    axes[1].tick_params(axis='x', rotation=45)

                    for bar, num in zip(bars, count):
                        axes[1].text(bar.get_x() + bar.get_width() / 2 - 0.1, num + 1, str(num), fontsize=10, color='black', ha='center')

                    plt.tight_layout()

                    # Displaying plot using streamlit
                    st.pyplot(fig)

                else:
                    st.write(f"No data found for game with title '{search_term}'.")

if __name__ == "__main__":
    main()
