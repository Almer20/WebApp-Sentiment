import streamlit as st
from st_pages import hide_pages
from time import sleep
import mysql.connector
import bcrypt
from streamlit_option_menu import option_menu
import importlib


st.set_page_config(layout="centered")
hide_pages(["main","option","Searching", "Review", "Upload", "Preprocessing", "labelling", "Klasifikasi", "tes", "searchingbeta", "Sentiment"])

with st.sidebar:
    selected = option_menu(
        menu_title=None,  # Hide the title
        options=["Searching", "Upload", "Review","Preprocessing","Labelling","Klasifikasi","Sentiment","Log Out"],
        icons=["search", "cloud-upload", "file-text-fill", "body-text","file-text","hourglass","code-square","box-arrow-in-left"],
        menu_icon="cast",
        default_index=0,
        # orientation="horizontal",
    )

def log_out():
    st.session_state["logged_in"] = False
    st.success("Logged out!")
    sleep(0.5)
    st.switch_page('main.py') 

if selected == "Searching":
    module = importlib.import_module('pages.1_🔎_Searching')
    module.main()
elif selected == "Upload":
    module = importlib.import_module('pages.2_📥_Upload')
    module.main()
elif selected == "Review":
    module = importlib.import_module('pages.3_📝_Review')
    module.main()
elif selected == "Preprocessing":
    module = importlib.import_module('pages.4_🧾_Preprocessing')
    module.main()
elif selected == "Labelling":
    module = importlib.import_module('pages.5_📒_labelling')
    module.main()
elif selected == "Klasifikasi":
    module = importlib.import_module('pages.6_📚_Klasifikasi')
    module.main()
elif selected == "Sentiment":
    module = importlib.import_module('pages.7_🗿_Sentiment')
    module.main()
elif selected == "Log Out":
    log_out()