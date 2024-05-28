import streamlit as st
import pandas as pd
import csv
import pymysql

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

# Function to insert data into MySQL database
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
    st.title("Upload Review Pengguna Steam")

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
        st.write(df)

        # Connect to MySQL database
        conn = connect_to_db()

        if conn:

            # Insert data into MySQL database
            insert_data(conn, df.to_dict('records'))

            st.success("Data successfully uploaded to MySQL database.")


if __name__ == "__main__":
    main()
