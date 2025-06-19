# Blog Agentic RAG

A Streamlit-based application that leverages LangChain, LangGraph, and Google Generative AI to perform Retrieval-Augmented Generation (RAG) on blog posts. The system processes blog content from a provided URL, creates a vector store for semantic search, and uses an agentic workflow to answer user queries about topics such as LLMs, LLM agents, prompt engineering, and adversarial attacks on LLMs.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The Blog Agentic RAG application is designed to analyze blog posts and provide intelligent, context-aware responses to user queries. It uses a vector store (Qdrant) to index blog content and a LangGraph-based agentic workflow to retrieve relevant information, assess document relevance, rewrite queries if needed, and generate accurate responses. The application is accessible via a user-friendly Streamlit web interface.

## Features
- **Blog Post Analysis**: Loads and processes blog content from a user-provided URL.
- **Semantic Search**: Uses Qdrant vector store and Google Generative AI embeddings for similarity-based retrieval.
- **Agentic Workflow**: Employs LangGraph to manage a pipeline that includes query rewriting, document grading, and response generation.
- **Streamlit UI**: Provides an intuitive interface for entering API keys, loading blog posts, and querying the system.
- **Context-Aware Responses**: Generates answers tailored to the content of the analyzed blog post.

## Directory Structure
```
ahmedelkassrawy-blog-agentic-rag/
├── README.md                     # Project documentation
├── requirements.txt              # Project dependencies
└── src/
    └── tools/
        ├── agent.py              # Core agent logic and LangGraph workflow
        └── app.py                # Streamlit application entry point
```

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ahmedelkassrawy/blog-agentic-rag.git
   cd ahmedelkassrawy-blog-agentic-rag
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**:
   - Provide your API keys for Qdrant and Google Generative AI. You can enter these via the Streamlit UI or set them directly in your environment:
     ```bash
     export QDRANT_URL="your-qdrant-url"
     export QDRANT_API_KEY="your-qdrant-api-key"
     export GOOGLE_API_KEY="your-google-api-key"
     ```

5. **Run the Application**:
   ```bash
   streamlit run src/tools/app.py
   ```

## Usage
1. **Launch the Application**:
   - Run the Streamlit command above to open the web interface in your browser.
2. **Enter API Keys**:
   - In the sidebar, input your Qdrant URL, Qdrant API Key, and Google API Key, then click "Done".
3. **Load a Blog Post**:
   - Enter the URL of a blog post (e.g., related to LLMs or prompt engineering) and click "Load Blog Post".
4. **Query the System**:
   - Enter a query (e.g., "What are the key points about prompt engineering in the blog?") and click "Run Query".
   - The system will process the query and display the response based on the blog content.
5. **Example**:
   - URL: `https://example.com/blog/llm-agents`
   - Query: `Explain the role of LLM agents in automation.`
   - Response: A concise answer derived from the blog content.

## Technologies Used
- **Python**: Core programming language.
- **LangChain**: For RAG pipeline and integration with Google Generative AI.
- **LangGraph**: For managing the agentic workflow.
- **Google Generative AI**: Powers embeddings and language model for query processing.
- **Qdrant**: Vector store for semantic search.
- **Streamlit**: Web interface for user interaction.
- **Pydantic**: For structured data validation.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure your code adheres to the existing style and includes relevant tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
