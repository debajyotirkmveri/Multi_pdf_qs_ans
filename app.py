import os
import shutil
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure the Google Generative AI
genai.configure(api_key=api_key)

# Create the generative model instance
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Users dictionary for demo purposes
users = {
    "user1": hash_password("password1"),
    "user2": hash_password("password2"),
}

# Function to extract text from multiple PDFs with document names and page numbers
def get_pdf_text_with_pages(pdf_paths):
    text_chunks_with_pages = []
    for pdf_path in pdf_paths:
        try:
            pdf_reader = PdfReader(pdf_path)
            doc_name = os.path.basename(pdf_path)
            for page_number, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                if text:
                    text_chunks_with_pages.append((text, page_number, doc_name))
        except FileNotFoundError as e:
            print(f"Error: {e}")
    return text_chunks_with_pages

# Split text into chunks, including document name and page numbers
def get_text_chunks_with_pages(text_chunks_with_pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks_with_pages = []
    for text, page_number, doc_name in text_chunks_with_pages:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            chunks_with_pages.append((chunk, page_number, doc_name))
    return chunks_with_pages

# Create and save vector store with document names and page numbers
def get_vector_store_with_pages(text_chunks_with_pages, index_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    texts, page_numbers, doc_names = zip(*text_chunks_with_pages)
    vector_store = FAISS.from_texts(texts, embedding=embeddings)
    vector_store.save_local(index_name)
    # Save page numbers and document names to a file for later retrieval
    with open("page_numbers_docs.pkl", "wb") as f:
        pickle.dump((page_numbers, doc_names), f)

# Function to get the conversational chain
def get_conversational_chain():
    prompt_template = """
        You are an expert at extracting information from PDFs! Excellent job on your inquiry!
        Answer the question in a concise and structured way using bullet points to ensure clarity and easy understanding. 
        Do not provide the answer in paragraph form. Use numbered lists for sub-points if applicable.   
        If the answer is not available in the provided context, simply state: "The answer is not available in the context."
        If the question seems to contain errors, incomplete information, or unclear wording, try to interpret it, and ask the user for clarification if needed.
        Guidelines for identifying errors or unclear questions:
        - If the question contains misspelled words or unclear grammar, suggest a corrected version and confirm with the user.
        - If the question seems incomplete or ambiguous, mention the missing information and ask the user for more specifics.
        Context:\n {context}\n
        Question: \n{question}\n
        
        Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Handle user input and get response with document names and pages
def user_input_with_page(user_question, index_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    new_db = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
    
    # Perform the similarity search
    search_results = new_db.similarity_search(user_question, return_scores=False)
    
    chain = get_conversational_chain()

    response = chain.invoke(
        {"input_documents": search_results, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]

# Function to compute TF-IDF vectors for a list of documents
def compute_tfidf_vectors(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer

# Find sentences with a similarity score above the threshold, including document names and pages
def find_matching_sentences(result, text_chunks_with_pages, threshold=0.3):
    # Flatten the list of text chunks into sentences, along with page numbers and document names
    sentences_with_pages = [(sentence, page_number, doc_name)
                            for text_chunk, page_number, doc_name in text_chunks_with_pages
                            for sentence in re.split(r'(?<=[.!?]) +', text_chunk)]
    
    # Separate sentences, pages, and document names
    sentences, pages, doc_names = zip(*sentences_with_pages)
    
    # Compute TF-IDF vectors for all sentences
    tfidf_matrix, vectorizer = compute_tfidf_vectors(sentences)
    
    # Split the result into sentences
    result_sentences = re.split(r'(?<=[.!?]) +', result)
    
    # Compute TF-IDF vectors for result sentences
    result_tfidf_matrix = vectorizer.transform(result_sentences)
    
    # Calculate cosine similarity and find matches
    cosine_similarities = cosine_similarity(result_tfidf_matrix, tfidf_matrix)
    
    # Store matching sentences with pages and document names
    matching_sentences_pages = []
    for idx, similarities in enumerate(cosine_similarities):
        matching_indices = [i for i, score in enumerate(similarities) if score > threshold]
        matching_pages_docs = [(pages[i], doc_names[i]) for i in matching_indices]
        if matching_pages_docs:
            matching_sentences_pages.append((result_sentences[idx], set(matching_pages_docs)))
        else:
            matching_sentences_pages.append((result_sentences[idx], {("Page not found", "Doc not found")}))
    
    return matching_sentences_pages

# Ensure 'temp' directory exists before saving uploaded files
def ensure_temp_directory():
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    return temp_dir

# Login system
def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        hashed_password = hash_password(password)
        if username in users and users[username] == hashed_password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.sidebar.success(f"Welcome {username}!")
        else:
            st.sidebar.error("Incorrect username or password")

# Logout system
def logout():
    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['username'] = None


# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Page for user to process the question-answering functionality
def main():
    st.set_page_config(page_title='Multi-PDF QA Chatbot', layout="wide", page_icon="📄")
    local_css("style1.css")
    # Threshold slider
    # threshold = st.sidebar.slider("Select Similarity Threshold", min_value=0.2, max_value=0.3, value=0.3, step=0.01)
    # Define the submit function
    def submit():
        user_question = st.session_state.user_question
        if user_question:
            input_tokens = model.count_tokens(user_question)
            result = user_input_with_page(user_question, index_name="faiss_index")
            output_tokens = model.count_tokens(result)
            if "The answer is not available in the context." in result:
                st.session_state['chat_history'].append({
                    'question': user_question,
                    'response': result,
                    'pages': None,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                })
            else:
                st.session_state['chat_history'].append({
                    'question': user_question,
                    'response': result,
                    'pages': find_matching_sentences(result, text_chunks_with_pages),
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                })
            st.session_state.user_question = ""

    # Check login status and render pages accordingly
    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown(
        f"""
        <style>
        .login-container {{
            background-image: url('https://static.wixstatic.com/media/9d7b99_dfcb8e88751c4cecb7ac677976976ec8~mv2.gif');
            background-size: cover;
            background-position: center;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }}  """, 
        unsafe_allow_html=True)
        #st.sidebar.image("https://static.wixstatic.com/media/9d7b99_dfcb8e88751c4cecb7ac677976976ec8~mv2.gif", use_column_width=True)
        login()
        st.markdown('</div>', unsafe_allow_html=True)
        return
    elif 'interface_loaded' not in st.session_state:
        st.session_state['interface_loaded'] = True
        st.write('<style> .logo-container { position: absolute; center: 10px; right: 10px; } </style>', unsafe_allow_html=True)
        logo_url = "https://assets.ey.com/content/dam/ey-sites/ey-com/en_gl/topics/innovation-realized/ey-ey-stacked-logo.jpg"
        st.markdown(f'<div class="logo-container"><img src="{logo_url}" alt="EY Logo" width="700"></div>', unsafe_allow_html=True)
        st.button("process", on_click=lambda: st.session_state.update({"main_page": True}))
        return
    
    elif st.session_state.get('main_page'):
        logout()
        st.header('Multi-PDF Content-Based Question Answering System')

        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        #threshold = st.sidebar.slider("Select Similarity Threshold", min_value=0.2, max_value=0.3, value=0.3, step=0.01)

        # Display chat history
        st.subheader("Chat History")
        for entry in st.session_state['chat_history']:
            st.write(f"🧑 User Question: {entry['question']} (Input tokens: {entry['input_tokens']})")
            st.write(f"🤖 Bot Answer: {entry['response']} (Output tokens: {entry['output_tokens']})")
            st.write("📖 Source:")
            if entry.get('pages') and "The answer is not available in the context." not in entry['response']:
                # Filter out "Page not found" and "Doc not found" entries
                # Convert to a set of unique document-page combinations
                unique_sources = {
                    (page, doc) for _, pages_docs_set in entry['pages']
                    for page, doc in pages_docs_set
                    if page != "Page not found" and doc != "Doc not found"
                }

                # Format and display each unique source
                valid_sources = [f"Document: {doc}, Page: {page}" for page, doc in unique_sources]
                
                if valid_sources:
                    st.write(", ".join(valid_sources))


        # Sidebar file uploadeAr
        uploaded_files = st.sidebar.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            # Ensure temp directory exists before saving
            temp_dir = "temp"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # Save uploaded files in the temporary directory
            saved_files = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_files.append(file_path)

            # Process the PDF files
            text_chunks_with_pages = get_pdf_text_with_pages(saved_files)

            if text_chunks_with_pages:
                chunks_with_pages = get_text_chunks_with_pages(text_chunks_with_pages)
                get_vector_store_with_pages(chunks_with_pages, index_name="faiss_index")

                st.sidebar.success("PDF files processed and indexed successfully.")

        # Question input area
        st.text_input("Ask a question:", key="user_question", on_change=submit)

if __name__ == '__main__':
    main()


