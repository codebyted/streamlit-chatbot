"""
Full RAG Chatbot with PDF Processing, OCR, ChromaDB, and OpenAI
Implements complete Retrieval-Augmented Generation pipeline
"""

import streamlit as st
import PyPDF2
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import os
import re
import uuid
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
import numpy as np
from datetime import datetime
from streamlit_jupyter import run_streamlit_in_jupyter

# ==================== CONFIGURATION ====================

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'index_built' not in st.session_state:
    st.session_state.index_built = False
if 'doc_stats' not in st.session_state:
    st.session_state.doc_stats = {}
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
if 'collection' not in st.session_state:
    st.session_state.collection = None

# ==================== HELPER FUNCTIONS ====================

def extract_text_from_pdf(pdf_file) -> Tuple[str, int, bool]:
    """
    Extract text from PDF, with OCR fallback for scanned documents
    Returns: (text, num_pages, used_ocr)
    """
    try:
        # Try text extraction first
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            text += page_text + "\n"

        # Check if extraction was successful (more than 20 chars per page average)
        avg_chars_per_page = len(text) / num_pages if num_pages > 0 else 0

        if avg_chars_per_page < 20:
            # Likely a scanned PDF, use OCR
            st.info("🔍 Detected scanned PDF. Using OCR...")
            return extract_text_with_ocr(pdf_file.getvalue(), num_pages), num_pages, True

        return text, num_pages, False

    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return "", 0, False

def extract_text_with_ocr(pdf_bytes: bytes, num_pages: int) -> str:
    """
    Extract text using OCR (Tesseract)
    """
    try:
        # Convert PDF pages to images
        images = convert_from_bytes(pdf_bytes)

        text = ""
        progress_bar = st.progress(0)

        for i, image in enumerate(images):
            # Update progress
            progress_bar.progress((i + 1) / num_pages)

            # OCR the image
            page_text = pytesseract.image_to_string(image)
            text += f"\n--- Page {i+1} ---\n{page_text}\n"

        progress_bar.empty()
        return text

    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return ""

def clean_text(text: str) -> str:
    """
    Clean extracted text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep periods and commas
    text = re.sub(r'[^\w\s.,;:!?-]', '', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Split text into overlapping chunks with metadata
    """
    words = text.split()
    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)

        # Try to estimate page number (rough approximation)
        page_num = (start // 500) + 1  # Assume ~500 words per page

        chunks.append({
            'id': str(uuid.uuid4()),
            'text': chunk_text,
            'chunk_index': chunk_id,
            'page_estimate': page_num,
            'word_count': len(chunk_words)
        })

        start += (chunk_size - overlap)
        chunk_id += 1

    return chunks

def get_embeddings_batch(texts: List[str], client, model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Get embeddings for multiple texts in batch
    """
    try:
        response = client.embeddings.create(
            input=texts,
            model=model
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        return []

def create_vector_database(chunks: List[Dict], client, embedding_model: str, persist_dir: str = "chroma_db"):
    """
    Create ChromaDB vector database
    """
    try:
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=persist_dir)

        # Delete old collection if exists
        try:
            chroma_client.delete_collection("documents")
        except:
            pass

        # Create new collection
        collection = chroma_client.create_collection(
            name="documents",
            metadata={"description": "Document chunks with embeddings"}
        )

        # Process chunks in batches
        batch_size = 50
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk['text'] for chunk in batch]

            status_text.text(f"Processing batch {i//batch_size + 1}/{total_batches}...")

            # Get embeddings
            embeddings = get_embeddings_batch(batch_texts, client, embedding_model)

            if not embeddings:
                continue

            # Add to collection
            collection.add(
                ids=[chunk['id'] for chunk in batch],
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=[{
                    'chunk_index': chunk['chunk_index'],
                    'page_estimate': chunk['page_estimate'],
                    'word_count': chunk['word_count']
                } for chunk in batch]
            )

            progress_bar.progress((i + batch_size) / len(chunks))

        progress_bar.empty()
        status_text.empty()

        return chroma_client, collection

    except Exception as e:
        st.error(f"Vector DB error: {str(e)}")
        return None, None

def retrieve_relevant_chunks(query: str, collection, client, embedding_model: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieve most relevant chunks for a query
    """
    try:
        # Get query embedding
        query_embedding = get_embeddings_batch([query], client, embedding_model)[0]

        # Query collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Format results
        chunks = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                chunks.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else 0,
                    'id': results['ids'][0][i]
                })

        return chunks

    except Exception as e:
        st.error(f"Retrieval error: {str(e)}")
        return []

def calculate_confidence(distance: float) -> float:
    """
    Convert distance to confidence score (0-100%)
    """
    # Lower distance = higher confidence
    # Typical distances range from 0 to 2
    confidence = max(0, min(100, (1 - distance / 2) * 100))
    return round(confidence, 2)

def generate_response(query: str, retrieved_chunks: List[Dict], client, model: str, exam_mode: bool = False) -> str:
    """
    Generate response using GPT with retrieved context
    """
    try:
        # Build context from chunks
        context = "\n\n".join([
            f"[Page ~{chunk['metadata']['page_estimate']}]: {chunk['text']}"
            for chunk in retrieved_chunks
        ])

        # Build system prompt
        system_prompt = f"""You are an AI assistant for the AIU Student Handbook. Your role is to provide accurate, helpful answers based ONLY on the retrieved document chunks provided.

CRITICAL RULES:
1. Use ONLY information from the retrieved chunks below
2. Always cite page numbers when providing information (e.g., "According to page 15...")
3. If the information is not in the retrieved chunks, say "I don't have that information in the current context"
4. Do NOT make up or invent information
5. Be concise but complete
{"6. Provide short, exam-friendly explanations" if exam_mode else "6. Provide detailed, comprehensive explanations"}

RETRIEVED CONTEXT:
{context}
"""

        # Generate response
        response = client.chat.completions.create(
            input=query,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating response: {str(e)}"

# ==================== MAIN APP ====================

def main():
    st.title("📚 RAG Document Assistant")
    st.markdown("*Retrieval-Augmented Generation Chatbot for AIU Student Handbook*")

    # Create two columns
    col1, col2 = st.columns([1, 2])

    # ==================== LEFT COLUMN: Configuration & Upload ====================
    with col1:
        st.header("⚙️ Configuration")

        # API Key
        api_key = st.text_input("OpenAI API Key", type="password", key="api_key")

        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API key to continue")
            return

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Model selection
        model = st.selectbox(
            "Select Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0
        )

        embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            index=0
        )

        # Chunking parameters
        st.subheader("Chunking Parameters")
        chunk_size = st.slider("Chunk Size (words)", 500, 2000, 1000, 100)
        overlap = st.slider("Overlap (words)", 0, 500, 200, 50)

        # Retrieval parameters
        top_k = st.slider("Number of Chunks to Retrieve", 1, 10, 5)

        # Exam mode
        exam_mode = st.checkbox("Exam Mode (concise answers)")

        st.markdown("---")

        # PDF Upload
        st.subheader("📄 Upload Document")
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

        if uploaded_file:
            if st.button("🔨 Build Index", type="primary"):
                with st.spinner("Processing PDF..."):
                    # Extract text
                    text, num_pages, used_ocr = extract_text_from_pdf(uploaded_file)

                    if text:
                        st.success(f"✅ Extracted text from {num_pages} pages" +
                                 (" using OCR" if used_ocr else ""))

                        # Clean text
                        text = clean_text(text)

                        # Chunk text
                        chunks = chunk_text(text, chunk_size, overlap)
                        st.info(f"📦 Created {len(chunks)} chunks")

                        # Create vector database
                        with st.spinner("Building vector database..."):
                            chroma_client, collection = create_vector_database(
                                chunks, client, embedding_model
                            )

                        if collection:
                            st.session_state.chroma_client = chroma_client
                            st.session_state.collection = collection
                            st.session_state.index_built = True
                            st.session_state.doc_stats = {
                                'num_pages': num_pages,
                                'num_chunks': len(chunks),
                                'used_ocr': used_ocr
                            }
                            st.success("✅ Index built successfully!")
                    else:
                        st.error("Failed to extract text from PDF")

        # Display stats
        if st.session_state.index_built:
            st.markdown("---")
            st.subheader("📊 Document Statistics")
            stats = st.session_state.doc_stats
            st.metric("Pages", stats['num_pages'])
            st.metric("Chunks", stats['num_chunks'])
            st.metric("OCR Used", "Yes" if stats['used_ocr'] else "No")

        st.markdown("---")

        # Preset questions
        st.subheader("💡 Quick Questions")
        preset_questions = [
            "What are the graduation requirements?",
            "What is the dress code policy?",
            "What are the housing rules?",
            "How do I apply for scholarships?",
            "What is the student code of conduct?"
        ]

        for question in preset_questions:
            if st.button(question, key=f"preset_{question}"):
                st.session_state.current_query = question

        # Contact info
        st.markdown("---")
        st.subheader("📞 Quick Contacts")
        if st.button("Dean of Students Contact"):
            st.session_state.current_query = "What is the Dean of Students contact information?"
        if st.button("Security Contact"):
            st.session_state.current_query = "What are the security office contacts?"

        # Reset
        if st.button("🔄 Reset Session"):
            st.session_state.messages = []
            st.session_state.index_built = False
            st.session_state.doc_stats = {}
            st.rerun()

    # ==================== RIGHT COLUMN: Chat Interface ====================
    with col2:
        st.header("💬 Chat Interface")

        if not st.session_state.index_built:
            st.info("👈 Please upload a document and build the index first")
            return

        # Display chat history
        chat_container = st.container()

        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

                    if message["role"] == "assistant" and "chunks" in message:
                        with st.expander("📑 View Retrieved Context"):
                            for i, chunk in enumerate(message["chunks"]):
                                confidence = calculate_confidence(chunk['distance'])
                                st.markdown(f"""
                                **Chunk {i+1}** (Page ~{chunk['metadata']['page_estimate']},
                                Confidence: {confidence}%)

                                {chunk['text'][:500]}...

                                ---
                                """)

        # Chat input
        if 'current_query' in st.session_state:
            query = st.session_state.current_query
            del st.session_state.current_query
        else:
            query = st.chat_input("Ask a question about the handbook...")

        if query:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})

            # Display user message
            with st.chat_message("user"):
                st.markdown(query)

            # Retrieve relevant chunks
            with st.spinner("🔍 Searching document..."):
                retrieved_chunks = retrieve_relevant_chunks(
                    query,
                    st.session_state.collection,
                    client,
                    embedding_model,
                    top_k
                )

            # Generate response
            with st.spinner("🤔 Generating answer..."):
                response = generate_response(
                    query,
                    retrieved_chunks,
                    client,
                    model,
                    exam_mode
                )

            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "chunks": retrieved_chunks
            })

            # Display assistant message
            with st.chat_message("assistant"):
                st.markdown(response)

                # Display retrieved context
                with st.expander("📑 View Retrieved Context"):
                    for i, chunk in enumerate(retrieved_chunks):
                        confidence = calculate_confidence(chunk['distance'])
                        st.markdown(f"""
                        **Chunk {i+1}** (Page ~{chunk['metadata']['page_estimate']},
                        Confidence: {confidence}%)

                        {chunk['text'][:500]}...

                        ---
                        """)

                # Feedback buttons
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✅ Correct", key=f"correct_{len(st.session_state.messages)}"):
                        st.success("Thanks for your feedback!")
                with col_b:
                    if st.button("❌ Incorrect", key=f"incorrect_{len(st.session_state.messages)}"):
                        st.info("Feedback recorded. We'll improve!")

            st.rerun()



if __name__ == "__main__":
    # Check if running in a Jupyter environment
    if os.getenv("COLAB_JUPYTER_SERVER_ROOT") or os.getenv("JPY_SESSION_NAME") or os.getenv("VSCODE_CWD"):
        run_streamlit_in_jupyter(main)
    else:
        main()
