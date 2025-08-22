import streamlit as st
from echoanimal import main as echoanimal_main
from wiki_taxonomy import main as wiki_taxonomy_main

# Set page configuration at the top of the main script
# st.set_page_config(page_title="Unified Animal App", layout="wide")

# Sidebar Navigation
st.sidebar.title("Choose an Application")
app_choice = st.sidebar.radio("Select App:", ["Animal Sound Translator", "Animal Taxonomy Classifier"])

# App Navigation Logic
if app_choice == "Animal Sound Translator":
    st.title("Animal Sound Translator")
    echoanimal_main()
elif app_choice == "Animal Taxonomy Classifier":
    st.title("Animal Taxonomy Classifier")
    wiki_taxonomy_main()
