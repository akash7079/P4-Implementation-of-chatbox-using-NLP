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
