import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline
import torch
from urllib.parse import urlparse
import re

# Loading the models
@st.cache_resource
def load_models():
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return qa_pipeline, model

qa_pipeline, model = load_models()

# Enhanced Page Configuration
st.set_page_config(
    page_title="AI News Q&A Assistant", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive styling
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Header Styles */
    .header-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 0;
    }
    
    /* Sidebar Styles */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2D1B69 0%, #11998e 100%);
        border-radius: 15px;
        padding: 1rem;
    }
    
    .sidebar-title {
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Input Field Styles */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid transparent;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }
    
    /* Answer Card Styles */
    .answer-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
    }
    
    .answer-title {
        color: #2D1B69;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .answer-content {
        color: #333;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .source-info {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .source-label {
        color: #667eea;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .source-text {
        color: #555;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        font-style: italic;
    }
    
    /* Checkbox Styles */
    .stCheckbox > label {
        color: white;
        font-weight: 500;
    }
    
    /* Success/Error Message Styles */
    .stSuccess, .stError {
        border-radius: 10px;
        border: none;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Loading Spinner Custom */
    .stSpinner {
        text-align: center;
        color: white;
    }
    
    /* Feature Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.scheme) and bool(parsed.netloc)

def fetch_article(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except requests.RequestException as e:
        st.error(f"Error fetching article from {url}: {e}")
        return ""

def embed_articles(articles):
    all_paragraphs = []
    all_embeddings = []
    article_sources = []
    
    for url, article in articles:
        if article.strip():
            paragraphs = article.split('\n')
            embeddings = model.encode(paragraphs)
            all_paragraphs.extend(paragraphs)
            all_embeddings.extend(embeddings)
            article_sources.extend([url] * len(paragraphs))
        
    return all_paragraphs, np.array(all_embeddings), article_sources

def create_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def get_answer(question, all_paragraphs, index, article_sources, max_sentences=3):
    question_embedding = model.encode([question])
    D, I = index.search(np.array(question_embedding), k=5)
    
    candidate_paragraphs = [all_paragraphs[i] for i in I[0]]
    candidate_sources = [article_sources[i] for i in I[0]]
    
    best_answer = None
    best_source = None
    best_score = 0
    best_paragraph = None
    
    for paragraph, source in zip(candidate_paragraphs, candidate_sources):
        if not paragraph.strip():
            continue
        result = qa_pipeline(question=question, context=paragraph)
        if result['score'] > best_score:
            best_score = result['score']
            best_answer = result['answer']
            best_source = source
            
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)
            best_paragraph = ' '.join(sentences[:max_sentences])
    
    return best_answer, best_source, best_paragraph

# Main Header
st.markdown("""
    <div class="header-container">
        <h1 class="main-title">🚀 AI News Q&A Assistant</h1>
        <p class="subtitle">Get instant answers from news articles using advanced AI technology</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with enhanced styling
st.sidebar.markdown('<h2 class="sidebar-title">📝 Input Center</h2>', unsafe_allow_html=True)

# URL inputs with better labeling
st.sidebar.markdown("### 🔗 News Article URLs")
url1 = st.sidebar.text_input('🌐 Article URL 1', placeholder="https://example.com/article1")
url2 = st.sidebar.text_input('🌐 Article URL 2', placeholder="https://example.com/article2")
url3 = st.sidebar.text_input('🌐 Article URL 3', placeholder="https://example.com/article3")

# Question input
st.sidebar.markdown("### ❓ Your Question")
question = st.sidebar.text_input('What would you like to know?', placeholder="e.g., What is the main topic of the article?")

# Options
st.sidebar.markdown("### ⚙️ Options")
show_source_paragraph = st.sidebar.checkbox('📖 Show Source Paragraph', value=True)

# Enhanced button
if st.sidebar.button('🔍 Get Answer', key="main_button"):
    if question:
        valid_urls = [url for url in [url1, url2, url3] if is_valid_url(url)]
        if valid_urls:
            with st.spinner('🤖 AI is analyzing articles and generating your answer...'):
                articles = [(url, fetch_article(url)) for url in valid_urls]
                all_paragraphs, all_embeddings, article_sources = embed_articles(articles)
                
                if all_paragraphs:
                    index = create_index(all_embeddings)
                    answer, source, best_paragraph = get_answer(question, all_paragraphs, index, article_sources)
                    if answer:
                        st.success('✅ Answer generated successfully!')
                        
                        # Enhanced answer display
                        st.markdown(f"""
                            <div class="answer-card">
                                <h3 class="answer-title">🎯 Answer</h3>
                                <div class="answer-content">{answer}</div>
                                
                                {f'''
                                <div class="source-info">
                                    <div class="source-label">📚 Source Context</div>
                                    <div class="source-text">{best_paragraph}</div>
                                </div>
                                ''' if show_source_paragraph else ''}
                                
                                <div class="source-info">
                                    <div class="source-label">🔗 Source URL</div>
                                    <div class="source-text"><a href="{source}" target="_blank">{source}</a></div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error('❌ No answer could be found in the provided articles.')
                else:
                    st.error('❌ No valid content was found in the provided articles.')
        else:
            st.error('❌ Please provide at least one valid URL.')
    else:
        st.error('❌ Please provide a question.')

# Feature showcase when no results are displayed
if 'main_button' not in st.session_state:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">AI-Powered Analysis</div>
                <div class="feature-desc">Advanced natural language processing to understand and answer your questions</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Lightning Fast</div>
                <div class="feature-desc">Get answers in seconds with optimized embedding and search algorithms</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Multiple Sources</div>
                <div class="feature-desc">Analyze up to 3 different news articles simultaneously for comprehensive answers</div>
            </div>
        """, unsafe_allow_html=True)

# Instructions
st.markdown("""
    <div class="answer-card">
        <h3 class="answer-title">📋 How to Use</h3>
        <div class="answer-content">
            <ol>
                <li><strong>Add URLs:</strong> Paste up to 3 news article URLs in the sidebar</li>
                <li><strong>Ask Questions:</strong> Enter your question about the articles</li>
                <li><strong>Get Answers:</strong> Click "Get Answer" to receive AI-generated responses</li>
                <li><strong>View Sources:</strong> Check the source paragraph option to see context</li>
            </ol>
        </div>
    </div>
""", unsafe_allow_html=True)
