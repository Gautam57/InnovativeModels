gorq_API_KEY_2 ="Enter_your_API_Key"
Youtube_Video_ID = "Enter_your_video_ID"
google_gemini_AI_API_KEY = "Enter_your_API_Key"

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load the GROQ and Google Gemini API keys
groq_api_key = gorq_API_KEY_2
os.environ["GOOGLE_API_KEY"] = google_gemini_AI_API_KEY

st.title("Gemma Model Video Q&A")

# Define the model
llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="Llama3-8b-8192")

# Define the prompt template that now accepts a single "context" variable
prompt_template = ChatPromptTemplate.from_template(
"""
Answer the questions based on the context provided below. Consider both the video transcript and user comments (positive, negative, and neutral) to provide an accurate and well-rounded answer.

<Context>
{context}

Questions: {input}
"""
)

def vector_embedding():
    if "vectors" not in st.session_state:
        # Use Google Generative AI Embeddings for the model
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load PDF documents from the folder
        folder_path = r'/Users/gautambr/Documents/Great Learning/Natural language Processing/YoutubeCommentAnalysis/Is Make In India Failing? | Case study'        
        # Load the video transcript and comments (positive, negative, neutral)
        st.session_state.loader = PyPDFDirectoryLoader(folder_path)
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        
        # Split the documents into chunks for easier processing
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # Split into final documents (limiting to 20 for the sake of example)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        
        # Create vector embeddings using FAISS
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# User input for asking questions
prompt1 = st.text_input("Enter Your Question About the Video")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time

if prompt1:
    # Extract and combine the contexts for video and comments into a single context string
    context_video = ""  # This will hold the text from the video transcript
    context_positive = ""  # Positive comments
    context_negative = ""  # Negative comments
    context_neutral = ""  # Neutral comments
    
    # Retrieve the relevant documents from the vector store (assuming you have vector retrieval logic in place)
    for doc in st.session_state.final_documents:
        if "video" in doc.metadata.get("source", "").lower():
            context_video += doc.page_content
        elif "positive" in doc.metadata.get("source", "").lower():
            context_positive += doc.page_content
        elif "negative" in doc.metadata.get("source", "").lower():
            context_negative += doc.page_content
        elif "neutral" in doc.metadata.get("source", "").lower():
            context_neutral += doc.page_content

    # Combine all contexts into a single context variable
    combined_context = f"""
    <Video Transcript Context>
    {context_video}

    <Positive Comments>
    {context_positive}

    <Negative Comments>
    {context_negative}

    <Neutral Comments>
    {context_neutral}
    """

    # Now, create the prompt using the combined context
    prompt_with_combined_context = prompt_template.format(context=combined_context, input=prompt1)

    # Create a retrieval chain with the combined context
    document_chain = create_stuff_documents_chain(llm, prompt_template)  # <-- Use prompt_template here
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    
    # Invoke the retrieval chain to get a response
    response = retrieval_chain.invoke({'input': prompt1})
    
    st.write("Response time: ", time.process_time() - start)
    st.write(response['answer'])
    
    # Show similar documents that were used for context (from the retrieval)
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")




