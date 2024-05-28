import streamlit as st
from st_pages import hide_pages
from time import sleep
import mysql.connector
import bcrypt
from streamlit_option_menu import option_menu
import importlib

# Connect to MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="webanalysisdemo"
)
cursor = conn.cursor()
st.set_page_config(layout="wide", initial_sidebar_state='collapsed')
def log_in(username, password):
    cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    if user:
        hashed_password = user[0]
        if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
            st.session_state["logged_in"] = True
            hide_pages([])  # No pages are hidden after login
            st.success("Logged in!")
            sleep(0.5)
            # st.experimental_rerun() # Reload the page
            st.switch_page('pages/option.py') 
        else:
            st.error("Incorrect username or password")
    else:
        st.error("User not found")

def register(username, email, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, hashed_password))
        conn.commit()
        st.success("Registration successful")
    except mysql.connector.Error as e:
        st.error(f"Error occurred: {e}")

def log_out():
    st.session_state["logged_in"] = False
    st.success("Logged out!")
    sleep(0.5)
    st.experimental_rerun()  # Reload the page

def show_login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Log in"):
        log_in(username, password)

def show_register():
    st.title("Register")
    new_username = st.text_input("New Username")
    new_email = st.text_input("Email")
    new_password = st.text_input("New Password", type="password")

    if st.button("Register"):
        register(new_username, new_email, new_password)
        
def show_searching():
    st.title("Searching")
    new_username = st.text_input("New Username")
    new_email = st.text_input("Email")
    new_password = st.text_input("New Password", type="password")

    if st.button("Register"):
        register(new_username, new_email, new_password)


def show_main():
    st.title("Welcome to the main app")
    st.sidebar.title("Main Menu")
    if st.sidebar.button("Log out"):
        log_out()

    # Add other main app functionality here

# Hide pages if not logged in
if not st.session_state.get("logged_in", False):
    hide_pages(["option","Searching", "Review", "Upload", "Preprocessing", "labelling", "Klasifikasi", "tes", "searchingbeta"])

    selected = option_menu(
        menu_title=None,  # Hide the title
        options=["Login", "Searching"],
        icons=["box-arrow-in-right", "person-plus", "search"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    if selected == "Login":
        show_login()
    # elif selected == "Register":
    #     show_register()
    elif selected == "Searching":
        module = importlib.import_module('pages.1_🔎_Searching')
        module.main()
else:
    show_main()
