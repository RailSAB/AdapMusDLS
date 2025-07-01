# AdapMusDLS - Adaptive Music Discovery and Logging System

A Telegram bot for music discovery using semantic similarity search with FAISS indexing and Weaviate vector database.

## Project Structure

### Dataset Processing (`dataset/`)

- **Data Source**: YouTube Creative Commons music dataset from [HuggingFace](https://huggingface.co/datasets/WaveGenAI/youtube-cc-by-music_annoted)
- **Processing Pipeline**: 
  - Data cleaning and preprocessing in `main.ipynb`
  - Embedding generation using SentenceTransformers
  - Dimensionality reduction with PCA for efficient storage
  - Cleaned dataset with reduced embeddings for similarity search

### Similarity Search (`similarity_search/`)

- **FAISS Index**: IndexIVFFlat with Inner Product (IP) distance for fast approximate nearest neighbor search
- **Index Configuration**: 
  - Uses inverted file structure with adaptive nlist (sqrt of dataset size)
  - L2 normalized embeddings with Inner Product similarity
  - Chosen for balance between speed and accuracy on medium-scale datasets
- **Index Generation**: `make_index.py` creates optimized FAISS index from embeddings
- **Similarity Engine**: `similarity.py` provides search functionality with:
  - Hybrid search combining semantic and lexical matching
  - Query expansion using WordNet synonyms
  - Integration with Weaviate vector database
- **Advantages**: Sub-linear search time, memory efficient, supports clustering-based search

### Telegram Bot (`tg/`)

- **Bot Implementation**: `main.py` handles user interactions and search requests
- **Logging System**: `search_logger.py` tracks user queries and system performance
- **Features**:
  - Natural language music search
  - Query processing and result filtering
  - User session management
  - Comprehensive logging for analytics

## Quick Start

1. Start Weaviate database:
   ```bash
   docker-compose up -d
   ```
   (Also run in dataset folder main.ipynb to fill database)

2. Run the Telegram bot:
   ```bash
   cd tg
   python3 main.py
   ```

## Dependencies

- FAISS for similarity search
- Weaviate for vector database
- SentenceTransformers for embeddings
- python-telegram-bot for bot functionality
- scikit-learn for dimensionality reduction