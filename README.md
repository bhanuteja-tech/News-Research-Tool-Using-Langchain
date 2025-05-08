# News Research Tool 📈

A powerful Streamlit web application that helps you research news articles by processing content from URLs and answering questions about the content. The tool offers two modes of operation: a full-featured mode with OpenAI integration and a simplified mode that works without an API key.

![News Research Tool Screenshot](Screenshot%20\(188\).png)

## Features

### Dual Operation Modes
- **OpenAI Mode**: Uses AI for advanced semantic search and question answering
- **Simplified Mode**: Works without an API key using basic keyword search

### Core Functionality
- Process content from multiple news article URLs
- Extract and analyze text from web pages
- Ask questions about the processed content
- View sources and references for answers
- Dark theme UI for comfortable reading

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/bhanuteja-tech/News-Research-Tool-Using-Langchain.git
   cd LLM PROJECT
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. (Optional) Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   Alternatively, you can enter your API key directly in the application's sidebar.

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501).

3. Enter the URLs of news articles you want to analyze in the sidebar.

4. (Optional) Enter your OpenAI API key in the sidebar for enhanced functionality.

5. Select your preferred mode (OpenAI or Simplified).

6. Click the "Process URLs" button to extract and process the content.

7. Once processing is complete, you can ask questions about the articles in the main section of the app.

## How It Works

### OpenAI Mode
1. **Data Loading**: Extracts content from the provided URLs
2. **Text Processing**: Splits content into manageable chunks
3. **Embedding Creation**: Uses OpenAI to convert text into vector representations
4. **Vector Storage**: Stores embeddings in a FAISS database for efficient retrieval
5. **AI-Powered Q&A**: Generates comprehensive answers to your questions using OpenAI

### Simplified Mode
1. **Data Loading**: Extracts content from the provided URLs
2. **Text Processing**: Splits content into manageable chunks
3. **Keyword Search**: Performs basic keyword matching on your queries
4. **Result Ranking**: Ranks results based on keyword relevance
5. **Source Attribution**: Shows where the information came from

## Getting an OpenAI API Key

For the best experience, we recommend using an OpenAI API key:

1. Go to [OpenAI's platform](https://platform.openai.com/signup)
2. Create an account or sign in
3. Navigate to the API section
4. Create a new API key
5. Copy the key and use it in the application

Note: OpenAI offers free credits for new users, and the costs for this type of application are typically very low.

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- FAISS vector database
- Internet connection to access news articles
- (Optional) OpenAI API key for enhanced functionality

## License

[MIT License](LICENSE)
