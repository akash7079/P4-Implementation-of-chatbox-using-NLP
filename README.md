
# **Intents-Based Chatbot using NLP**

## **Project Overview**

This repository contains the code for a chatbot built using Natural Language Processing (NLP) techniques and machine learning. The chatbot is designed to interact with users in a conversational manner, recognize intents, and provide relevant responses. The project leverages machine learning models like Logistic Regression and utilizes text processing techniques like TF-IDF (Term Frequency-Inverse Document Frequency) for intent classification.

## **Features**

- **Intent-based Responses**: The chatbot can classify user input and respond accordingly based on predefined intents.
- **NLP Preprocessing**: Tokenization, lemmatization, and vectorization using TF-IDF are applied to prepare the data for analysis.
- **Entity Extraction**: Key medical or general terms like symptoms, conditions, or medications are identified using Named Entity Recognition (NER).
- **Conversation History**: The system logs user interactions and allows for easy retrieval of conversation history.
- **Dynamic and Rule-based Responses**: The chatbot can provide both predefined and dynamically generated responses based on user queries.
  
## **Technologies Used**

- **Python**: Programming language for the chatbot logic.
- **Streamlit**: Front-end framework for creating a simple user interface.
- **Scikit-learn**: Used for building and training machine learning models (Logistic Regression and TF-IDF vectorizer).
- **NLTK**: A library for natural language processing tasks like tokenization and lemmatization.
- **CSV**: For storing and managing conversation history.
  
## **Installation**

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- pip (Python package manager)

### Steps to Install

1. Clone the repository:
   ```bash
      git clone <repository-url> cd <repository-directory>wing installed:
2. Create a virtual enviorment:
   ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install required packages:
   ```bash
     pip install -r requirements.txt
4. Download NLTK Data
   ```bash
    import nltk
    nltk.download('punkt')

## **Usage**
To run the chatbot application, execute the following command:

## **streamlit run app.py**
Once the application is running, you can interact with the chatbot through the web interface. Type your message in the input box and press Enter to see the chatbot's response.

## **Intents Data**
The chatbot's behavior is defined by the intents.json file, which contains various tags, patterns, and responses. You can modify this file to add new intents or change existing ones.

## **Conversation History**
The chatbot saves the conversation history in a CSV file (chat_log.csv). You can view past interactions by selecting the "Conversation History" option in the sidebar.

## **Contributing**
Contributions to this project are welcome! If you have suggestions for improvements or features, feel free to open an issue or submit a pull request.

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.

## **Acknowledgments**
- NLTK for natural language processing.
- Scikit-learn for machine learning algorithms.
- Streamlit for building the web interface.
