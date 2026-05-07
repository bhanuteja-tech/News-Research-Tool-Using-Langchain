import os
import streamlit as st
import pickle
import requests
import re
from collections import Counter
from urllib.parse import urlparse
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from bs4 import BeautifulSoup
from langchain.schema import Document

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)


def _is_openrouter_key(api_key: str) -> bool:
    return bool(api_key) and api_key.strip().startswith("sk-or-")


def _openai_client_kwargs():
    """
    Build OpenAI-compatible client kwargs for both OpenAI and OpenRouter.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    api_base = (
        os.environ.get("OPENAI_API_BASE", "").strip()
        or os.environ.get("OPENAI_BASE_URL", "").strip()
    )

    if _is_openrouter_key(api_key) and not api_base:
        api_base = "https://openrouter.ai/api/v1"

    kwargs = {"openai_api_key": api_key}
    if api_base:
        kwargs["openai_api_base"] = api_base

    return kwargs


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


def _extract_source_url(doc):
    """
    Return a normalized source URL for a document chunk.
    """
    if not hasattr(doc, "metadata") or not isinstance(doc.metadata, dict):
        return "Unknown source"
    return (
        doc.metadata.get("source")
        or doc.metadata.get("url")
        or doc.metadata.get("source_url")
        or "Unknown source"
    )


def _merge_missing_urls_with_fallback(data, valid_urls):
    """
    Ensure content exists for as many input URLs as possible.
    If primary loader misses some sources, fetch only the missing URLs via fallback.
    """
    loaded_sources = set()
    for doc in data:
        src = _extract_source_url(doc)
        if src and src != "Unknown source":
            loaded_sources.add(src)

    missing_urls = [u for u in valid_urls if u not in loaded_sources]
    if not missing_urls:
        return data, []

    fallback_docs, failed_urls = _extract_text_with_fallback(missing_urls)
    if fallback_docs:
        data.extend(fallback_docs)
    return data, failed_urls


def _select_diverse_docs(retrieved_docs, max_docs=8, max_per_source=2):
    """
    Pick relevant chunks while preventing one URL from dominating context.
    """
    if not retrieved_docs:
        return []

    selected = []
    per_source = {}

    for doc in retrieved_docs:
        source = _extract_source_url(doc)
        current = per_source.get(source, 0)
        if current >= max_per_source:
            continue
        selected.append(doc)
        per_source[source] = current + 1
        if len(selected) >= max_docs:
            break

    return selected


def _looks_like_non_article_url(url):
    """
    Heuristic check for homepage/section URLs that usually produce noisy context.
    """
    try:
        parsed = urlparse(url)
        path = (parsed.path or "").strip("/").lower()
        if not path:
            return True
        non_article_paths = {
            "news", "latest", "top-news", "world", "business", "markets", "sports",
            "technology", "tech", "politics", "india", "us", "uk",
        }
        return path in non_article_paths and path.count("/") == 0
    except Exception:
        return False


def _load_expected_sources_from_session_or_disk():
    """
    Return URL sources that should be represented in retrieval.
    """
    expected = st.session_state.get("last_processing_debug", {}).get("loaded_sources", []) or []
    if expected:
        return expected

    if os.path.exists("processed_urls.txt"):
        try:
            with open("processed_urls.txt", "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except Exception:
            return []
    return []


def _retrieve_with_source_coverage(vectorstore, query, expected_sources, per_source_k=2, total_k=8):
    """
    Retrieve chunks by enforcing per-source coverage first, then fill remaining slots.
    """
    selected = []
    seen_ids = set()
    covered_sources = []

    for source in expected_sources:
        try:
            per_source_docs = vectorstore.similarity_search(
                query,
                k=per_source_k,
                filter={"source": source},
            )
        except Exception:
            per_source_docs = []

        for doc in per_source_docs:
            doc_id = f"{_extract_source_url(doc)}::{hash(doc.page_content[:300])}"
            if doc_id in seen_ids:
                continue
            selected.append(doc)
            seen_ids.add(doc_id)

        if per_source_docs:
            covered_sources.append(source)

    if len(selected) < total_k:
        fallback_docs = vectorstore.max_marginal_relevance_search(query, k=24, fetch_k=64)
        for doc in fallback_docs:
            doc_id = f"{_extract_source_url(doc)}::{hash(doc.page_content[:300])}"
            if doc_id in seen_ids:
                continue
            selected.append(doc)
            seen_ids.add(doc_id)
            if len(selected) >= total_k:
                break

    return selected[:total_k], covered_sources


def _generate_suggested_questions(urls, docs, n=5):
    """
    Build a small set of user-facing sample questions from processed content.
    """
    stopwords = {
        "about", "after", "again", "also", "article", "articles", "because", "being", "between",
        "could", "first", "from", "have", "into", "just", "more", "most", "news", "other",
        "over", "that", "their", "there", "these", "they", "this", "those", "through",
        "under", "very", "what", "when", "where", "which", "while", "with", "would",
    }
    keyword_counter = Counter()
    domain_hints = []

    for url in urls:
        try:
            host = urlparse(url).netloc.replace("www.", "")
            if host and host not in domain_hints:
                domain_hints.append(host)
        except Exception:
            continue

    for doc in docs[:20]:
        text = getattr(doc, "page_content", "")[:2500].lower()
        tokens = re.findall(r"[a-z][a-z\-]{3,}", text)
        for token in tokens:
            if token not in stopwords and not token.isdigit():
                keyword_counter[token] += 1

    top_keywords = [w for w, _ in keyword_counter.most_common(4)]
    keyword_a = top_keywords[0] if len(top_keywords) > 0 else "topic"
    keyword_b = top_keywords[1] if len(top_keywords) > 1 else "developments"
    keyword_c = top_keywords[2] if len(top_keywords) > 2 else "impact"

    questions = [
        f"What are the main points discussed across these articles?",
        f"How do these articles explain {keyword_a} and {keyword_b}?",
        f"What recent developments are mentioned, and what could happen next?",
        f"What are the key risks, opportunities, or implications related to {keyword_c}?",
        "Where do the articles agree, and where do they present different views?",
    ]

    if domain_hints:
        questions[2] = f"What are the major updates reported by sources like {', '.join(domain_hints[:2])}?"

    return questions[:n]


def _load_saved_suggestions():
    """
    Rebuild suggestions from saved processed artifacts after reruns/restarts.
    """
    urls = []
    docs = []
    try:
        if os.path.exists("processed_urls.txt"):
            with open("processed_urls.txt", "r", encoding="utf-8") as f:
                urls = [line.strip() for line in f.readlines() if line.strip()]
    except Exception:
        urls = []

    try:
        if os.path.exists("processed_docs.pkl"):
            with open("processed_docs.pkl", "rb") as f:
                docs = pickle.load(f)
    except Exception:
        docs = []

    if docs:
        return _generate_suggested_questions(urls, docs)
    return []

# Initialize session state early so sidebar logic can safely read it
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'simplified_mode' not in st.session_state:
    st.session_state.simplified_mode = not os.environ.get("OPENAI_API_KEY")
if 'suggested_questions' not in st.session_state:
    st.session_state.suggested_questions = []
if 'last_processing_debug' not in st.session_state:
    st.session_state.last_processing_debug = {}
if 'last_retrieval_debug' not in st.session_state:
    st.session_state.last_retrieval_debug = {}

# Recover suggestions on reruns so users always see question ideas after processing.
if not st.session_state.get("suggested_questions"):
    st.session_state.suggested_questions = _load_saved_suggestions()

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
        # OpenRouter keys need OpenAI-compatible base URL.
        if _is_openrouter_key(api_key) and not os.environ.get("OPENAI_API_BASE"):
            os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

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
file_path = "faiss_index"

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
                non_article_urls = [u for u in valid_urls if _looks_like_non_article_url(u)]

                if not valid_urls:
                    st.error("No valid URLs provided. Please enter at least one valid URL.")
                else:
                    if non_article_urls:
                        st.warning(
                            "Some URLs look like home/section pages and may reduce answer relevance: "
                            + ", ".join(non_article_urls[:3])
                            + (" ..." if len(non_article_urls) > 3 else "")
                            + ". Prefer direct article URLs."
                        )
                    status_text.markdown("<div class='info-box'>Step 1/4: Loading data from URLs...</div>", unsafe_allow_html=True)
                    # Load data
                    try:
                        loader = UnstructuredURLLoader(urls=valid_urls)
                        data = loader.load()
                        data, failed_urls = _merge_missing_urls_with_fallback(data, valid_urls)
                        if failed_urls:
                            st.warning(
                                "Could not extract content from: "
                                + ", ".join(failed_urls[:3])
                                + (" ..." if len(failed_urls) > 3 else "")
                            )
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
                            loaded_sources = []
                            for d in data:
                                src = _extract_source_url(d)
                                if src not in loaded_sources and src != "Unknown source":
                                    loaded_sources.append(src)
                            progress_bar.progress(25)

                            status_text.markdown("<div class='info-box'>Step 2/4: Splitting text into chunks...</div>", unsafe_allow_html=True)
                            # Split data
                            text_splitter = RecursiveCharacterTextSplitter(
                                separators=['\n\n', '\n', '.', ','],
                                chunk_size=1000
                            )
                            docs = text_splitter.split_documents(data)
                            st.session_state.last_processing_debug = {
                                "input_urls": valid_urls,
                                "loaded_sources": loaded_sources,
                                "failed_urls": failed_urls,
                                "total_docs": len(docs),
                            }
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
                                st.session_state.suggested_questions = _generate_suggested_questions(valid_urls, docs)
                            else:
                                status_text.markdown("<div class='info-box'>Step 3/4: Creating embeddings...</div>", unsafe_allow_html=True)
                                # Create embeddings using OpenAI
                                embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
                                embeddings = OpenAIEmbeddings(
                                    model=embedding_model,
                                    **_openai_client_kwargs()
                                )
                                vectorstore_openai = FAISS.from_documents(docs, embeddings)
                                progress_bar.progress(75)

                                status_text.markdown("<div class='info-box'>Step 4/4: Saving knowledge base...</div>", unsafe_allow_html=True)
                                # Save the FAISS index using native serializer (pickle can fail on locks)
                                vectorstore_openai.save_local(file_path)
                                # Save docs for generating question suggestions
                                with open("processed_docs.pkl", "wb") as f:
                                    pickle.dump(docs, f)
                                st.session_state.suggested_questions = _generate_suggested_questions(valid_urls, docs)
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
                            loaded_sources = []
                            for d in data:
                                src = _extract_source_url(d)
                                if src not in loaded_sources and src != "Unknown source":
                                    loaded_sources.append(src)
                            st.session_state.last_processing_debug = {
                                "input_urls": valid_urls,
                                "loaded_sources": loaded_sources,
                                "failed_urls": failed_urls,
                                "total_docs": len(docs),
                            }
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
                                st.session_state.suggested_questions = _generate_suggested_questions(valid_urls, docs)
                            else:
                                status_text.markdown("<div class='info-box'>Step 3/4: Creating embeddings...</div>", unsafe_allow_html=True)
                                embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
                                embeddings = OpenAIEmbeddings(
                                    model=embedding_model,
                                    **_openai_client_kwargs()
                                )
                                vectorstore_openai = FAISS.from_documents(docs, embeddings)
                                progress_bar.progress(75)

                                status_text.markdown("<div class='info-box'>Step 4/4: Saving knowledge base...</div>", unsafe_allow_html=True)
                                vectorstore_openai.save_local(file_path)
                                with open("processed_docs.pkl", "wb") as f:
                                    pickle.dump(docs, f)
                                st.session_state.suggested_questions = _generate_suggested_questions(valid_urls, docs)
                                progress_bar.progress(100)

                            status_text.markdown("<div class='success-box'>✅ Processing complete! You can now ask questions about the articles.</div>", unsafe_allow_html=True)
                            st.session_state.processing_complete = True
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Divider
    st.markdown("---")

    # Debug information panel
    with st.expander("Debug: URL ingestion and retrieval"):
        processing_debug = st.session_state.get("last_processing_debug", {})
        retrieval_debug = st.session_state.get("last_retrieval_debug", {})

        if not processing_debug and not retrieval_debug:
            st.markdown("No debug data yet. Process URLs and ask a question to populate this panel.")
        else:
            if processing_debug:
                st.markdown("**Last processing run**")
                input_urls = processing_debug.get("input_urls", [])
                loaded_sources = processing_debug.get("loaded_sources", [])
                failed_urls = processing_debug.get("failed_urls", [])
                total_docs = processing_debug.get("total_docs", 0)

                st.markdown(f"- Input URLs: {len(input_urls)}")
                st.markdown(f"- Sources loaded: {len(loaded_sources)}")
                st.markdown(f"- Total chunks created: {total_docs}")

                if input_urls:
                    st.markdown("Input URLs:")
                    for u in input_urls:
                        st.markdown(f"- {u}")

                if loaded_sources:
                    st.markdown("Loaded sources:")
                    for s in loaded_sources:
                        st.markdown(f"- {s}")

                if failed_urls:
                    st.markdown("Failed URLs:")
                    for f in failed_urls:
                        st.markdown(f"- {f}")

            if retrieval_debug:
                st.markdown("**Last answer retrieval**")
                query_text = retrieval_debug.get("query", "")
                retrieved_sources = retrieval_debug.get("retrieved_sources", [])
                raw_hits = retrieval_debug.get("raw_hits", 0)
                selected_hits = retrieval_debug.get("selected_hits", 0)
                expected_sources = retrieval_debug.get("expected_sources", [])
                covered_sources = retrieval_debug.get("covered_sources", [])

                if query_text:
                    st.markdown(f"- Query: `{query_text}`")
                st.markdown(f"- Retrieved chunks (after targeted retrieval + fill): {raw_hits}")
                st.markdown(f"- Chunks sent to model: {selected_hits}")
                if expected_sources:
                    st.markdown(f"- Expected sources: {len(expected_sources)}")
                if covered_sources:
                    st.markdown(f"- Sources covered by targeted retrieval: {len(covered_sources)}")

                if retrieved_sources:
                    st.markdown("Sources used in answer:")
                    for s in retrieved_sources:
                        st.markdown(f"- {s}")

    # Q&A Section
    st.markdown("<h2 class='subheader'>Ask Questions About the Articles</h2>", unsafe_allow_html=True)

    # Example questions
    with st.expander("See example questions"):
        suggested_questions = st.session_state.get("suggested_questions", [])
        if suggested_questions:
            st.markdown("Suggested questions based on processed URLs:")
            for i, q in enumerate(suggested_questions, start=1):
                st.markdown(f"- {q}")
        else:
            st.markdown("""
            - What are the main topics discussed in these articles?
            - What are the key insights about [specific topic]?
            - How does [company/person] relate to the news?
            - What are the implications of [event] mentioned in the articles?
            - What evidence or sources are cited most often?
            """)

    # Question input
    suggested_questions = st.session_state.get("suggested_questions", [])
    if suggested_questions:
        st.markdown("**Try one of these questions:**")
        selected_question = st.selectbox(
            "Suggested questions",
            options=[""] + suggested_questions,
            index=0,
            key="selected_suggested_question"
        )
        default_query = selected_question if selected_question else "What are the main points in these articles?"
    else:
        default_query = "What are the main points in these articles?"

    query = st.text_input("Enter your question:", value=default_query, placeholder="What are the main points in these articles?")

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
                        # Recreate embeddings and load FAISS index from disk
                        embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
                        embeddings = OpenAIEmbeddings(
                            model=embedding_model,
                            **_openai_client_kwargs()
                        )
                        vectorstore = FAISS.load_local(
                            file_path,
                            embeddings,
                            allow_dangerous_deserialization=True
                        )

                        # Initialize the LLM
                        default_model = "openai/gpt-4o-mini" if _is_openrouter_key(os.environ.get("OPENAI_API_KEY", "")) else "gpt-4o-mini"
                        llm_model = os.environ.get("OPENAI_MODEL", default_model)
                        llm = ChatOpenAI(
                            model=llm_model,
                            temperature=0.7,
                            max_tokens=500,
                            **_openai_client_kwargs()
                        )

                        # Retrieve top relevant chunks and ask the chat model directly.
                        # This avoids provider-specific chain parsing issues with OpenRouter.
                        expected_sources = _load_expected_sources_from_session_or_disk()
                        retrieved_docs, covered_sources = _retrieve_with_source_coverage(
                            vectorstore,
                            query,
                            expected_sources=expected_sources,
                            per_source_k=2,
                            total_k=8,
                        )
                        raw_retrieved_count = len(retrieved_docs)
                        context_blocks = []
                        source_urls = []
                        for i, doc in enumerate(retrieved_docs, start=1):
                            source_url = _extract_source_url(doc)
                            context_blocks.append(f"[Source {i} | {source_url}]\n{doc.page_content}")
                            if source_url and source_url != "Unknown source":
                                source_urls.append(source_url)
                        unique_retrieved_sources = []
                        for src in source_urls:
                            if src not in unique_retrieved_sources:
                                unique_retrieved_sources.append(src)
                        st.session_state.last_retrieval_debug = {
                            "query": query,
                            "raw_hits": raw_retrieved_count,
                            "selected_hits": len(retrieved_docs),
                            "retrieved_sources": unique_retrieved_sources,
                            "expected_sources": expected_sources,
                            "covered_sources": covered_sources,
                        }

                        prompt = (
                            "You are a helpful research assistant. Answer the question using only the context below.\n"
                            "Rules:\n"
                            "1) Give a concise direct answer first.\n"
                            "2) Then provide a short 'By source' section.\n"
                            "3) If a source is missing relevant evidence, say so explicitly.\n"
                            "4) Do not invent facts outside context.\n\n"
                            f"Question: {query}\n\n"
                            "Context:\n"
                            + "\n\n".join(context_blocks)
                        )
                        ai_response = llm.invoke(prompt)
                        answer_text = ai_response.content if hasattr(ai_response, "content") else str(ai_response)

                        # Display the answer with a visual separator
                        answer_cols = st.columns([1, 3, 1])
                        with answer_cols[1]:
                            st.markdown("<h3>Answer:</h3>", unsafe_allow_html=True)
                            st.markdown(f"<div class='info-box'>{answer_text}</div>", unsafe_allow_html=True)

                            # Add a decorative element
                            st.markdown("<div style='text-align: center; margin: 20px 0;'>✨✨✨</div>", unsafe_allow_html=True)

                        # Display sources
                        unique_sources = []
                        for src in source_urls:
                            if src not in unique_sources:
                                unique_sources.append(src)
                        if unique_sources:
                            with answer_cols[1]:
                                st.markdown("<h3>Sources:</h3>", unsafe_allow_html=True)
                                st.markdown("<div class='info-box' style='background-color: #1a2a3a;'>", unsafe_allow_html=True)
                                for source in unique_sources:
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
