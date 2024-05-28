import streamlit as st
from streamlit_option_menu import option_menu

def display_sidebar():
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=["Searching", "Review", "Upload", "Preprocessing", "Labelling", "Klasifikasi", "Tes", "Searching Beta", "Log Out"],
            icons=["search", "star", "upload", "gear", "tag", "list", "check-circle", "search", "box-arrow-right"],
            menu_icon="cast",
            default_index=0,
        )

    return selected

if __name__ == "__main__":
    main()
