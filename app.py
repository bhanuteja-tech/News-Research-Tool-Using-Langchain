import os
import streamlit as st
import pickle
import requests
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from bs4 import BeautifulSoup
from langchain.schema import Document

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)


def _extract_text_with_fallback(urls):
    """
    Robust URL text extraction for deployments where UnstructuredURLLoader
    can fail due to anti-bot checks or dynamic markup.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    extracted_docs = []
    failed_urls = []

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove non-content noise early
            for tag in soup(["script", "style", "noscript", "svg", "footer", "nav", "aside"]):
                tag.decompose()

            # Prefer common article containers first
            preferred_selectors = [
                "article",
                "main",
                '[role="main"]',
                ".article-body",
                ".story-body",
                ".post-content",
                ".entry-content",
            ]
            text = ""
            for selector in preferred_selectors:
                node = soup.select_one(selector)
                if node:
                    text = node.get_text(separator=" ", strip=True)
                    if len(text) > 300:
                        break

            # Fallback to entire page text when article container is missing
            if len(text) <= 300:
                text = soup.get_text(separator=" ", strip=True)

            # Normalize whitespace and keep useful content only
            text = " ".join(text.split())
            if len(text) > 300:
                extracted_docs.append(Document(page_content=text, metadata={"source": url}))
            else:
                failed_urls.append(url)
        except Exception:
            failed_urls.append(url)

    return extracted_docs, failed_urls

# Initialize session state early so sidebar logic can safely read it
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'simplified_mode' not in st.session_state:
    st.session_state.simplified_mode = not os.environ.get("OPENAI_API_KEY")

# Page configuration
st.set_page_config(
    page_title="News Research Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --background-color: #0e1117;
        --text-color: #f0f2f6;
        --primary-color: #4c8bf5;
        --secondary-color: #2e3440;
        --accent-color: #4c8bf5;
    }

    /* Override Streamlit's default white background */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* Header styles */
    .main-header {
        font-size: 2.5rem;
        color: var(--primary-color);
    }

    .subheader {
        font-size: 1.5rem;
        color: var(--text-color);
    }

    /* Box styles */
    .info-box {
        background-color: #1e2130;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary-color);
    }

    .success-box {
        background-color: #1e2d24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #00c853;
    }

    .warning-box {
        background-color: #2d2c1e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ffab00;
    }

    .error-box {
        background-color: #2d1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ff5252;
    }

    /* Remove white background from text inputs */
    .stTextInput > div > div > input {
        background-color: #1e2130;
        color: var(--text-color);
    }

    /* Style the sidebar */
    .css-1d391kg {
        background-color: #1a1c25;
    }

    /* Style expander */
    .streamlit-expanderHeader {
        background-color: #1e2130;
        color: var(--text-color);
    }

    /* Style buttons */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
    }

    /* Remove white background from containers */
    div.stBlock {
        background-color: transparent !important;
    }

    /* Style links */
    a {
        color: var(--primary-color);
    }

    /* Style progress bar */
    .stProgress > div > div > div > div {
        background-color: var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='main-header'>News Research Tool 📈</h1>", unsafe_allow_html=True)

# No header image

st.markdown("""
<div class='info-box'>
This tool helps you research news articles by:
<ol>
    <li>Processing content from multiple URLs</li>
    <li>Creating a knowledge base using AI embeddings</li>
    <li>Answering your questions based on the processed content</li>
</ol>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("<h2 class='subheader'>News Article URLs</h2>", unsafe_allow_html=True)
    st.markdown("Enter URLs of news articles to process and analyze:")

    # URL inputs with validation
    urls = []
    for i in range(3):
        url = st.text_input(f"URL {i+1}", key=f"url_{i}")
        if url:
            clean_url = url.strip()
            # Simple URL validation
            if not clean_url.startswith(('http://', 'https://')):
                st.warning(f"URL {i+1} should start with http:// or https://")
            else:
                urls.append(clean_url)

    # API Key input
    api_key = st.text_input("OpenAI API Key (optional)", type="password",
                           help="Enter your OpenAI API key if not set in .env file")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # Mode selection
    st.markdown("---")
    st.markdown("<h3>App Mode</h3>", unsafe_allow_html=True)

    # Only show mode selection if API key is available
    if os.environ.get("OPENAI_API_KEY"):
        mode_options = ["OpenAI Mode (Better results)", "Simplified Mode (No API key)"]
        selected_mode = st.radio(
            "Select Mode:",
            mode_options,
            index=0 if not st.session_state.get("simplified_mode", False) else 1
        )

        # Update session state based on selection
        st.session_state.simplified_mode = (selected_mode == mode_options[1])
    else:
        st.info("Running in Simplified Mode (No API key)")
        st.session_state.simplified_mode = True

    # Process button
    process_url_clicked = st.button("Process URLs", type="primary")

    # About section
    st.markdown("---")
    st.markdown("<h3>About</h3>", unsafe_allow_html=True)

    if st.session_state.simplified_mode:
        st.markdown("""
        Simplified Mode uses:
        - Basic keyword search
        - No OpenAI API key required
        - Limited question answering
        """)
    else:
        st.markdown("""
        OpenAI Mode uses:
        - LangChain for document processing
        - OpenAI for embeddings and Q&A
        - FAISS for vector storage
        """)

# Main content
file_path = "vector_index.pkl"

# Main content area
main_container = st.container()

with main_container:
    # Process URLs
    if process_url_clicked:
        if not urls or all(url == "" for url in urls):
            st.error("Please enter at least one valid URL")
        else:
            # Respect selected mode; only force simplified if OpenAI mode is selected without key
            if not st.session_state.simplified_mode and not os.environ.get("OPENAI_API_KEY"):
                st.warning("OpenAI API key is missing. Falling back to simplified mode.")
                st.session_state.simplified_mode = True

            try:
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Filter out empty URLs
                valid_urls = [url for url in urls if url.strip() != ""]

                if not valid_urls:
                    st.error("No valid URLs provided. Please enter at least one valid URL.")
                else:
                    status_text.markdown("<div class='info-box'>Step 1/4: Loading data from URLs...</div>", unsafe_allow_html=True)
                    # Load data
                    try:
                        loader = UnstructuredURLLoader(urls=valid_urls)
                        data = loader.load()
                        if not data:
                            # Retry with a fallback HTML extraction path
                            data, failed_urls = _extract_text_with_fallback(valid_urls)
                            if failed_urls:
                                st.warning(
                                    "Could not extract content from: "
                                    + ", ".join(failed_urls[:3])
                                    + (" ..." if len(failed_urls) > 3 else "")
                                )
                        if not data:
                            st.error("Could not extract any content from the provided URLs. Please try direct article links and avoid home/topic pages.")
                        else:
                            progress_bar.progress(25)

                            status_text.markdown("<div class='info-box'>Step 2/4: Splitting text into chunks...</div>", unsafe_allow_html=True)
                            # Split data
                            text_splitter = RecursiveCharacterTextSplitter(
                                separators=['\n\n', '\n', '.', ','],
                                chunk_size=1000
                            )
                            docs = text_splitter.split_documents(data)
                            progress_bar.progress(50)

                            # Check if we're in simplified mode
                            if st.session_state.simplified_mode:
                                status_text.markdown("<div class='info-box'>Step 3/4: Processing text (simplified mode)...</div>", unsafe_allow_html=True)
                                # In simplified mode, we'll just save the documents directly
                                progress_bar.progress(75)

                                status_text.markdown("<div class='info-box'>Step 4/4: Saving processed data...</div>", unsafe_allow_html=True)
                                # Save the documents to a pickle file
                                with open("processed_docs.pkl", "wb") as f:
                                    pickle.dump(docs, f)
                                # Also save the URLs for reference
                                with open("processed_urls.txt", "w") as f:
                                    f.write("\n".join(valid_urls))
                                progress_bar.progress(100)
                                # Set the file path for the simplified mode
                                st.session_state.simplified_file_path = "processed_docs.pkl"
                            else:
                                status_text.markdown("<div class='info-box'>Step 3/4: Creating embeddings...</div>", unsafe_allow_html=True)
                                # Create embeddings using OpenAI
                                embeddings = OpenAIEmbeddings()
                                vectorstore_openai = FAISS.from_documents(docs, embeddings)
                                progress_bar.progress(75)

                                status_text.markdown("<div class='info-box'>Step 4/4: Saving knowledge base...</div>", unsafe_allow_html=True)
                                # Save the FAISS index
                                with open(file_path, "wb") as f:
                                    pickle.dump(vectorstore_openai, f)
                                progress_bar.progress(100)

                            # Success message
                            status_text.markdown("<div class='success-box'>✅ Processing complete! You can now ask questions about the articles.</div>", unsafe_allow_html=True)
                            st.session_state.processing_complete = True
                    except Exception as e:
                        # If primary loader fails (common in cloud deployments), use fallback extraction
                        data, failed_urls = _extract_text_with_fallback(valid_urls)
                        if not data:
                            st.error(f"Error loading URLs: {str(e)}")
                            st.info("Tips: Make sure the URLs are accessible and point to text-based content (not PDFs or images).")
                        else:
                            if failed_urls:
                                st.warning(
                                    "Some URLs could not be extracted: "
                                    + ", ".join(failed_urls[:3])
                                    + (" ..." if len(failed_urls) > 3 else "")
                                )
                            progress_bar.progress(25)

                            status_text.markdown("<div class='info-box'>Step 2/4: Splitting text into chunks...</div>", unsafe_allow_html=True)
                            text_splitter = RecursiveCharacterTextSplitter(
                                separators=['\n\n', '\n', '.', ','],
                                chunk_size=1000
                            )
                            docs = text_splitter.split_documents(data)
                            progress_bar.progress(50)

                            if st.session_state.simplified_mode:
                                status_text.markdown("<div class='info-box'>Step 3/4: Processing text (simplified mode)...</div>", unsafe_allow_html=True)
                                progress_bar.progress(75)

                                status_text.markdown("<div class='info-box'>Step 4/4: Saving processed data...</div>", unsafe_allow_html=True)
                                with open("processed_docs.pkl", "wb") as f:
                                    pickle.dump(docs, f)
                                with open("processed_urls.txt", "w") as f:
                                    f.write("\n".join(valid_urls))
                                progress_bar.progress(100)
                                st.session_state.simplified_file_path = "processed_docs.pkl"
                            else:
                                status_text.markdown("<div class='info-box'>Step 3/4: Creating embeddings...</div>", unsafe_allow_html=True)
                                embeddings = OpenAIEmbeddings()
                                vectorstore_openai = FAISS.from_documents(docs, embeddings)
                                progress_bar.progress(75)

                                status_text.markdown("<div class='info-box'>Step 4/4: Saving knowledge base...</div>", unsafe_allow_html=True)
                                with open(file_path, "wb") as f:
                                    pickle.dump(vectorstore_openai, f)
                                progress_bar.progress(100)

                            status_text.markdown("<div class='success-box'>✅ Processing complete! You can now ask questions about the articles.</div>", unsafe_allow_html=True)
                            st.session_state.processing_complete = True
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Divider
    st.markdown("---")

    # Q&A Section
    st.markdown("<h2 class='subheader'>Ask Questions About the Articles</h2>", unsafe_allow_html=True)

    # Example questions
    with st.expander("See example questions"):
        st.markdown("""
        - What are the main topics discussed in these articles?
        - What are the key insights about [specific topic]?
        - How does [company/person] relate to the news?
        - What are the implications of [event] mentioned in the articles?
        """)

    # Question input
    query = st.text_input("Enter your question:", placeholder="What are the main points in these articles?")

    if query:
        # Check if we need to use simplified mode or OpenAI mode
        simplified_mode = st.session_state.get('simplified_mode', False)
        simplified_file_path = st.session_state.get('simplified_file_path', 'processed_docs.pkl')

        if simplified_mode and not os.path.exists(simplified_file_path):
            st.warning("Please process some URLs first before asking questions.")
        elif not simplified_mode and not os.path.exists(file_path):
            st.warning("Please process some URLs first before asking questions.")
        else:
            try:
                with st.spinner("Searching for relevant information..."):
                    if simplified_mode:
                        # Simplified mode - basic keyword search without OpenAI
                        with open(simplified_file_path, "rb") as f:
                            docs = pickle.load(f)

                        # Simple keyword search in the documents
                        query_terms = query.lower().split()
                        matching_docs = []

                        for doc in docs:
                            content = doc.page_content.lower()
                            score = sum(1 for term in query_terms if term in content)
                            if score > 0:
                                matching_docs.append((doc, score))

                        # Sort by relevance score
                        matching_docs.sort(key=lambda x: x[1], reverse=True)

                        # Take top 3 results
                        top_results = matching_docs[:3] if matching_docs else []

                        # Display the results
                        answer_cols = st.columns([1, 3, 1])
                        with answer_cols[1]:
                            st.markdown("<h3>Search Results:</h3>", unsafe_allow_html=True)

                            if not top_results:
                                st.markdown("<div class='info-box'>No matching content found. Try different search terms.</div>", unsafe_allow_html=True)
                            else:
                                for i, (doc, score) in enumerate(top_results):
                                    st.markdown(f"<h4>Result {i+1}:</h4>", unsafe_allow_html=True)
                                    st.markdown(f"<div class='info-box'>{doc.page_content[:500]}...</div>", unsafe_allow_html=True)
                                    if hasattr(doc, 'metadata') and doc.metadata.get('source'):
                                        st.markdown(f"<small>Source: {doc.metadata.get('source')}</small>", unsafe_allow_html=True)

                            # Add a note about simplified mode
                            st.markdown("<div style='text-align: center; margin: 20px 0;'>⚠️ Running in simplified mode without OpenAI API. For better results, add your API key.</div>", unsafe_allow_html=True)
                    else:
                        # OpenAI mode - use the vector store and LLM
                        # Load the vector store
                        with open(file_path, "rb") as f:
                            vectorstore = pickle.load(f)

                        # Initialize the LLM
                        llm = OpenAI(temperature=0.7, max_tokens=500)

                        # Create the chain
                        chain = RetrievalQAWithSourcesChain.from_llm(
                            llm=llm,
                            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
                        )

                        # Get the answer
                        result = chain({"question": query}, return_only_outputs=True)

                        # Display the answer with a visual separator
                        answer_cols = st.columns([1, 3, 1])
                        with answer_cols[1]:
                            st.markdown("<h3>Answer:</h3>", unsafe_allow_html=True)
                            st.markdown(f"<div class='info-box'>{result['answer']}</div>", unsafe_allow_html=True)

                            # Add a decorative element
                            st.markdown("<div style='text-align: center; margin: 20px 0;'>✨✨✨</div>", unsafe_allow_html=True)

                        # Display sources
                        sources = result.get("sources", "")
                        if sources:
                            with answer_cols[1]:
                                st.markdown("<h3>Sources:</h3>", unsafe_allow_html=True)
                                st.markdown("<div class='info-box' style='background-color: #1a2a3a;'>", unsafe_allow_html=True)
                                sources_list = sources.split("\n")
                                for source in sources_list:
                                    if source.strip():
                                        st.markdown(f"- {source}")
                                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                error_message = str(e)
                simplified_mode = st.session_state.get('simplified_mode', False)

                if simplified_mode:
                    st.error(f"An error occurred in simplified mode: {error_message}")
                    st.info("Troubleshooting tips for simplified mode: \n"
                           "1. Check your internet connection\n"
                           "2. Try processing fewer or smaller articles\n"
                           "3. Try using simpler search terms\n"
                           "4. For better results, add your OpenAI API key")
                else:
                    if "API key" in error_message.lower() or "apikey" in error_message.lower():
                        st.error("OpenAI API key error. Please check that your API key is valid and has sufficient credits.")
                        # Offer to switch to simplified mode
                        if st.button("Switch to Simplified Mode (No API Key Required)"):
                            st.session_state.simplified_mode = True
                            st.experimental_rerun()
                    elif "timeout" in error_message.lower() or "connection" in error_message.lower():
                        st.error("Network error. Please check your internet connection and try again.")
                    else:
                        st.error(f"An error occurred while processing your question: {error_message}")

                    # Provide troubleshooting tips
                    st.info("Troubleshooting tips: \n"
                           "1. Make sure your OpenAI API key is valid and has available credits\n"
                           "2. Check your internet connection\n"
                           "3. Try processing fewer or smaller articles\n"
                           "4. Try asking a simpler question")

# Add a footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("<h4>About</h4>", unsafe_allow_html=True)
    st.markdown("This app uses AI to analyze news articles and answer questions about them.")
with footer_col2:
    st.markdown("<h4>Technologies</h4>", unsafe_allow_html=True)
    st.markdown("Built with Streamlit, LangChain, OpenAI, and FAISS.")
with footer_col3:
    st.markdown("<h4>Resources</h4>", unsafe_allow_html=True)
    st.markdown("[Streamlit](https://streamlit.io/) • [LangChain](https://langchain.com/) • [OpenAI](https://openai.com/)")
