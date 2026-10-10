# agentic-chatbot

## Streamlit chat UI

The Streamlit interface connects directly to the existing multi-agent
LangGraph workflow. It keeps separate conversations in the current browser
session and shows which agent handled each reply.

1. Install the project dependencies:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. Make sure Ollama is running and the configured models are available. The
   defaults are `qwen3.5:4b` for routing and `qwen3.5:latest` for agents.
3. Add `.pdf` or `.txt` files to `data/documents/`, then ingest them into the
   Chroma vector store:

   ```bash
   python3 -m app.rag.ingestion
   ```

   Ingestion splits the documents into chunks and creates their embeddings in
   `data/chroma_db/`. Run it again when you add documents.
4. Start the interface from the repository root:

   ```bash
   python3 -m streamlit run streamlit_app.py
   ```

Configuration can be customized with the environment variables documented in
[`app/config.py`](app/config.py), or in a local `.env` file.
