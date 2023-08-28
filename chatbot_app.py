import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Load the admission data from the CSV file
data = pd.read_csv('admission_data - Sheet1.csv')

# Extract queries and responses from the DataFrame
admission_queries = data['Query'].tolist()
admission_responses = data['Intent'].tolist()

# Tokenize and preprocess the admission queries
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = tokenizer.tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    return ' '.join(tokens)

preprocessed_queries = [preprocess_text(query) for query in admission_queries]

# Create a TF-IDF vectorizer and transform the queries
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_queries)

# Function to handle user queries
def handle_admission_query(user_query):
    preprocessed_user_query = preprocess_text(user_query)
    tfidf_user_query = vectorizer.transform([preprocessed_user_query])

    # Calculate cosine similarities between user query and admission queries
    similarities = cosine_similarity(tfidf_user_query, tfidf_matrix)

    # Find the most similar admission query
    most_similar_index = similarities.argmax()
    if similarities[0][most_similar_index] > 0.2:
        return admission_responses[most_similar_index]
    else:
        return "I'm sorry, I don't have information about that specific query."

# Streamlit UI
st.title("Admission Chatbot")

user_query = st.text_input("Enter your question:")
if st.button("Ask"):
    response = handle_admission_query(user_query)
    st.text("Chatbot: " + response)
