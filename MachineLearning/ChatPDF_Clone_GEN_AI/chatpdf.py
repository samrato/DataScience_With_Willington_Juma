import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
import tempfile

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(page_title="PDF Q&A Chat", page_icon="📄", layout="wide")
    st.header("💬 Chat with your PDF")
    
    # Sidebar with instructions and info
    with st.sidebar:
        st.title("About")
        st.markdown("""
        ### How to use:
        1. Upload a PDF file
        2. Wait for processing
        3. Ask questions about the content
        
        ### Example questions:
        - "Summarize this document"
        - "What are the main points?"
        - "Explain section about [topic]"
        - "List the key findings"
        """)
        
        # Display API status
        if os.getenv('OPENAI_API_KEY'):
            st.success("✅ API Key Loaded")
        else:
            st.error("❌ API Key Missing")
            st.info("Add OPENAI_API_KEY to your .env file")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "processed_file" not in st.session_state:
        st.session_state.processed_file = None

    # File upload
    pdf_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
    
    # Process PDF when uploaded
    if pdf_file is not None:
        # Check if this is a new file
        current_file_name = pdf_file.name
        if (st.session_state.processed_file != current_file_name or 
            st.session_state.vector_store is None):
            
            with st.spinner("📖 Reading and processing PDF..."):
                # Extract text from PDF
                text = ""
                pdf_reader = PdfReader(pdf_file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    text += page_text
                
                if not text.strip():
                    st.error("❌ Could not extract text from PDF. Please try a different file.")
                    return
                
                # Show PDF info
                st.success(f"✅ PDF loaded: {len(pdf_reader.pages)} pages")
                
                # Split text
                text_splitter = CharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
                
                # Create vector store
                try:
                    embeddings = OpenAIEmbeddings()
                    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
                    st.session_state.processed_file = current_file_name
                    st.success(f"📚 Document processed: {len(chunks)} sections ready for questions!")
                    
                except Exception as e:
                    st.error(f"❌ Error creating embeddings: {str(e)}")
                    return
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if question := st.chat_input("Ask a question about your PDF..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Find similar chunks
                        similar_docs = st.session_state.vector_store.similarity_search(question, k=3)
                        context = "\n".join([doc.page_content for doc in similar_docs])
                        
                        # Use ChatOpenAI
                        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                        
                        prompt = f"""
                        Based on the following context from a PDF document, provide a comprehensive answer to the question.
                        
                        Context:
                        {context}
                        
                        Question: {question}
                        
                        Instructions:
                        - Answer based only on the provided context
                        - If the context doesn't contain relevant information, say "I cannot find this information in the document"
                        - Be precise and helpful
                        
                        Answer:
                        """
                        
                        response = llm.invoke(prompt)
                        
                        # Display response
                        st.markdown(response.content)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response.content})
                    
                    except Exception as e:
                        error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                        st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    else:
        # Welcome message
        st.info("👆 Please upload a PDF file to start chatting!")
        
        # Sample questions when no PDF is uploaded
        st.markdown("""
        ### Once you upload a PDF, you can ask questions like:
        - **Summary Questions:** "What is this document about?", "Summarize the key points"
        - **Specific Content:** "What does it say about [specific topic]?", "Explain section 3"
        - **Data Extraction:** "List all the recommendations", "What statistics are mentioned?"
        - **Comparative Questions:** "Compare the different approaches mentioned"
        """)

if __name__ == "__main__":
    main()