# dashboard.py
import streamlit as st

def show_dashboard(user_id):
    st.title("Dashboard")
    # Add your dashboard content here

    def preprocessing():
        st.title("Preprocessing")
        st.write("prep")

    def visualize():
        st.title("Visualize")
        st.write("Vis")

    def main():
        st.sidebar.title("Navigation")
        selection = st.sidebar.radio("Go to", ["Preprocessing", "Visualize"])

        if selection == "Preprocessing":
            preprocessing()
        elif selection == "Visualize":
            visualize()

    if __name__ == "__main__":
        main()
