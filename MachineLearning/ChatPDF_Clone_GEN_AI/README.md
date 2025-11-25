# **LangChain PDF Chat Application - Complete Documentation**

## **Table of Contents**
1. [Introduction to LangChain](#introduction-to-langchain)
2. [Project Overview](#project-overview)
3. [Step-by-Step Installation](#step-by-step-installation)
4. [Detailed Component Explanations](#detailed-component-explanations)
5. [Complete Code Implementation](#complete-code-implementation)
6. [API Setup Guide](#api-setup-guide)
7. [Troubleshooting](#troubleshooting)
8. [Cost Optimization](#cost-optimization)

---

## **Introduction to LangChain**

### **What is LangChain?**
LangChain is a framework designed to simplify the development of applications using large language models (LLMs). It provides:

- **Standardized Interfaces** - Common interface for different LLM providers
- **Chain Components** - Connect multiple LLM calls in sequences
- **Memory Management** - Maintain conversation history
- **Document Loaders** - Process various file formats (PDF, CSV, etc.)
- **Vector Stores** - Efficient similarity search for documents

### **Why Use LangChain for PDF Chat?**
- **Document Processing**: Handles PDF text extraction and chunking
- **Semantic Search**: Finds relevant content using embeddings
- **Context Management**: Maintains conversation context
- **Multi-Provider Support**: Works with OpenAI, Google PaLM, HuggingFace, etc.

---

## **Project Overview**

### **How It Works**
1. **PDF Upload** → User uploads PDF document
2. **Text Extraction** → Extract text from PDF pages
3. **Text Chunking** → Split text into manageable pieces
4. **Embedding Generation** → Convert text to numerical vectors
5. **Vector Storage** → Store embeddings for quick retrieval
6. **Question Processing** → Convert user question to embedding
7. **Similarity Search** → Find most relevant text chunks
8. **Answer Generation** → LLM generates answer from context

### **Architecture Flow**
```
PDF File → Text Extraction → Text Chunking → Embeddings → Vector Store
                                                         ↓
User Question → Embedding → Similarity Search → Context + Question → LLM → Answer
```

---

## **Step-by-Step Installation**

### **1. Create Project Structure**
```bash
# Create project directory
mkdir langchain_pdf_chat
cd langchain_pdf_chat

# Create virtual environment (Python 3.8+ recommended)
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### **2. Upgrade pip**
```bash
pip install --upgrade pip
```

### **3. Install Dependencies One-by-One**

#### **Core Framework**
```bash
pip install langchain
```
*Purpose: Main LangChain framework for LLM operations*

#### **Web Interface**
```bash
pip install streamlit
```
*Purpose: Creates web UI for PDF upload and chat interface*

#### **PDF Processing**
```bash
pip install PyPDF2
```
*Purpose: Extracts text from PDF files*

#### **Vector Database**
```bash
pip install faiss-cpu --default-timeout=200
```
*Purpose: Stores and searches text embeddings efficiently*
*Note: `--default-timeout=200` prevents timeout during download*

#### **API Clients**
```bash
pip install openai
pip install google-generativeai
```
*Purpose: Connect to OpenAI GPT and Google PaLM APIs*

#### **Environment Management**
```bash
pip install python-dotenv
```
*Purpose: Securely manage API keys in .env file*

### **4. Verify Installation**
```bash
# Test each package
python -c "import langchain; print('LangChain OK')"
python -c "import streamlit; print('Streamlit OK')"
python -c "import PyPDF2; print('PyPDF2 OK')"
python -c "import faiss; print('FAISS OK')"
python -c "import openai; print('OpenAI OK')"
python -c "import google.generativeai; print('Google AI OK')"
```

---

## **Detailed Component Explanations**

### **1. PyPDF2 - PDF Text Extraction**
```python
from PyPDF2 import PdfReader

pdf_reader = PdfReader(pdf_file)
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()
```
- **Purpose**: Reads PDF files and extracts textual content
- **How it works**: Iterates through each page and extracts text
- **Limitations**: May not handle scanned PDFs (images)

### **2. CharacterTextSplitter - Text Chunking**
```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_text(text)
```
- **chunk_size**: Maximum characters per chunk (1000 = optimal for context)
- **chunk_overlap**: Overlap between chunks (preserves context)
- **Why chunk?**: LLMs have token limits; chunking enables processing large documents

### **3. OpenAIEmbeddings - Vector Conversion**
```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks, embeddings)
```
- **Embeddings**: Convert text to numerical vectors
- **Semantic Meaning**: Similar texts have similar vector representations
- **FAISS**: Facebook AI Similarity Search - efficient vector database

### **4. FAISS - Similarity Search**
```python
docs = knowledge_base.similarity_search(user_question)
```
- **How it works**: Compares question embedding with document embeddings
- **Returns**: Most relevant text chunks for the question
- **Efficiency**: Handles thousands of vectors quickly

### **5. Load QA Chain - Answer Generation**
```python
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)  # Lower temperature = more deterministic
chain = load_qa_chain(llm, chain_type="stuff")
response = chain.run(input_documents=docs, question=user_question)
```
- **Chain Type "stuff"**: Stuff all relevant chunks into LLM context
- **Temperature**: Controls randomness (0 = consistent answers)
- **Process**: LLM generates answer using provided context

---

## **Complete Code Implementation**

### **File Structure**
```
langchain_pdf_chat/
├── venv/                    # Virtual environment
├── .env                     # API keys (NOT in version control)
├── app.py                   # Main Streamlit application
├── requirements.txt         # Dependencies list
└── README.md               # This documentation
```

### **Main Application (app.py)**
```python
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(page_title="Chat with PDF", page_icon="📚")
    st.header("Chat with your PDF 💬")
    
    # PDF upload
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # Extract text from PDF
    text = ""
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None returns
        
        if not text.strip():
            st.error("No text could be extracted from the PDF. It might be a scanned document.")
            return
        
        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # Show user input
        user_question = st.text_input("Ask a question about your PDF:")
        
        if user_question:
            with st.spinner("Searching for answer..."):
                # Find similar chunks
                docs = knowledge_base.similarity_search(user_question, k=3)
                
                # Initialize LLM
                llm = OpenAI(temperature=0)
                chain = load_qa_chain(llm, chain_type="stuff")
                
                # Get answer with cost tracking
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question)
                    st.write(f"**Cost:** ${cb.total_cost:.4f}")
                
            st.write("**Answer:**")
            st.write(response)

if __name__ == '__main__':
    main()
```

### **Environment File (.env)**
```env
# OpenAI API Key
OPENAI_API_KEY=sk-your-openai-key-here

# Google PaLM API Key (Optional)
GOOGLE_API_KEY=your-google-key-here
```

### **Requirements File (requirements.txt)**
```txt
langchain==0.0.347
streamlit==1.28.0
PyPDF2==3.0.1
faiss-cpu==1.7.4
openai==0.28.1
python-dotenv==1.0.0
google-generativeai==0.3.0
```

---

## **API Setup Guide**

### **OpenAI API Key**
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up/login to your account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)
5. Add to `.env` as `OPENAI_API_KEY=sk-...`

### **Google PaLM API Key** (Free Alternative)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Click "Create API key"
4. Copy the key (starts with `AIzaSy...`)
5. Add to `.env` as `GOOGLE_API_KEY=AIzaSy...`

### **Using Google PaLM Instead of OpenAI**
```python
# Replace OpenAI with Google PaLM
from langchain.llms import GooglePalm

llm = GooglePalm(temperature=0, google_api_key=os.getenv('GOOGLE_API_KEY'))
```

---

## **Troubleshooting**

### **Common Installation Issues**

#### **FAISS Installation Fails**
```bash
# Try alternative installation methods
pip install faiss-cpu --no-cache-dir --timeout 300
# OR use ChromaDB instead
pip install chromadb
```
```python
# Replace FAISS with Chroma
from langchain.vectorstores import Chroma
knowledge_base = Chroma.from_texts(chunks, embeddings)
```

#### **LangChain Version Conflicts**
```bash
# Uninstall and reinstall specific version
pip uninstall langchain langchain-core langchain-community -y
pip install langchain==0.0.347
```

#### **PDF Text Extraction Issues**
```python
# Handle empty text extraction
text = page.extract_text() or ""
if not text.strip():
    st.error("This appears to be a scanned PDF. Please use a text-based PDF.")
```

### **API Key Issues**
- **Error**: "Invalid API Key" → Check key format and environment variables
- **Error**: "Rate Limit Exceeded" → Wait or upgrade plan
- **Error**: "Insufficient Funds" → Add billing to OpenAI account

---

## **Cost Optimization**

### **OpenAI Cost Management**
- **Embeddings**: ~$0.0004/1K tokens
- **GPT-3.5**: ~$0.002/1K tokens
- **Monitor Usage**: Use `get_openai_callback()` to track costs

### **Free Alternatives**
1. **Google PaLM API** - Free tier available
2. **Local Models** - Use HuggingFace models offline
3. **Lite Version** - Cache embeddings to avoid reprocessing

### **Cost-Saving Tips**
```python
# Cache the knowledge base to avoid reprocessing
@st.cache_resource
def create_knowledge_base(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embeddings)

# Use smaller chunks for cheaper processing
text_splitter = CharacterTextSplitter(
    chunk_size=500,  # Smaller chunks = cheaper
    chunk_overlap=100
)
```

---

## **Running the Application**

### **Start the App**
```bash
streamlit run app.py
```

### **Usage Steps**
1. Open browser to `http://localhost:8501`
2. Upload PDF document
3. Wait for processing completion
4. Ask questions about the PDF content
5. View answers and cost information

### **Expected Output**
- **PDF Processing**: Text extraction and chunking status
- **Question Answering**: Relevant answers with source context
- **Cost Tracking**: Real-time API usage costs
- **Error Handling**: Clear messages for invalid PDFs or API issues

---

## **Advanced Features to Add**

### **1. Conversation History**
```python
# Add to session state
if 'history' not in st.session_state:
    st.session_state.history = []

st.session_state.history.append((user_question, response))
```

### **2. Multiple PDF Support**
```python
# Process multiple PDFs
knowledge_bases = []
for pdf in uploaded_pdfs:
    # Process each PDF and combine knowledge bases
```

### **3. Source Citation**
```python
# Show which chunks were used
st.write("**Sources:**")
for i, doc in enumerate(docs):
    st.write(f"{i+1}. {doc.page_content[:100]}...")
```

This comprehensive documentation covers everything from basic concepts to advanced implementation, ensuring you understand each component and can troubleshoot effectively.
pip install langchain==0.0.230 openai

# Uninstall existing packages
pip uninstall langchain -y
pip uninstall langchain-openai -y
pip uninstall langchain-community -y

# Install the latest versions
pip install langchain
pip install langchain-openai
pip install langchain-community

pip uninstall langchain langchain-core langchain-community langchain-openai -y
pip install langchain langchain-community langchain-openai faiss-cpu openai python-dotenv streamlit PyPDF2