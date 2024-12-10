import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st  
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

def clear_history():
    if os.path.exists('chat_log.csv'):
        os.remove('chat_log.csv')
        st.success("Chat history cleared!")

counter = 0

def main():
    global counter
    st.set_page_config(page_title="Chatbot App", layout="wide")
    
    
    # Header
    st.title("Intents-based Chatbot using NLP")
    st.markdown("---")
    
    # Sidebar options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.header("Welcome to the Chatbot!")
        st.markdown("### Start Chatting")
        st.info("Ask your queries and get instant responses from the bot.")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("Type your message:", key=f"user_input_{counter}")

        if user_input:
            response = chatbot(user_input)
            st.markdown(f"**<span style='color:blue'>You:</span>** {user_input}", unsafe_allow_html=True)
            st.markdown(f"**<span style='color:green'>Chatbot:</span>** {response}", unsafe_allow_html=True)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header
                for row in csv_reader:
                    st.write(f"**You**: {row[0]}")
                    st.write(f"**Chatbot**: {row[1]}")
                    st.write(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.warning("No conversation history found.")

        if st.button("Clear Chat History"):
            clear_history()

    # About Menu
    elif choice == "About":
        st.header("About the Project")
        st.markdown("""
        This project demonstrates a chatbot built using **Natural Language Processing (NLP)** techniques and 
        the **Streamlit** framework. The chatbot is designed to respond to user queries based on pre-defined 
        intents and patterns.
        """)
        st.subheader("Features:")
        st.write("- Intent-based responses")
        st.write("- NLP preprocessing with TF-IDF")
        st.write("- Logistic Regression for classification")
        st.write("- Streamlit interface for user interaction")
        st.markdown("---")
        st.markdown("**Developed using Python and Streamlit.**")

if __name__ == '__main__':
    main()
