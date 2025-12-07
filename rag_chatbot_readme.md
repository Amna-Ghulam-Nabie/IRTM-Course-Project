# RAG Chatbot Project – README

## Project Title
Contextual Answering with RAG 

## Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot that combines classical IR techniques (BM25), dense retrieval using embeddings, cross-encoder re-ranking, and OpenAI's GPT model for generating context-aware responses. The system is capable of retrieving relevant information from a dataset of IR documents and providing accurate, natural language answers.

## Dataset
- `IR_dataset_entries.csv` contains the main corpus with the following columns:
  - `id`: Unique document identifier
  - `content`: Full text of the document
  - `title`: Title of the document
  - `category`: Topic category
  - `source`: Source of the document (e.g., research paper, blog)
- `evaluation_set.csv`: Contains a small evaluation set with queries and their relevant document IDs for computing metrics.

## Features
- BM25-based keyword retrieval
- Dense retrieval using SentenceTransformers embeddings
- Cross-Encoder for ranking candidate documents
- OpenAI GPT-4o-mini for generating final answers
- Interactive chatbot loop for testing queries
- Evaluation metrics (Precision@5, Recall@5, MRR, nDCG)

## Installation
1. Clone the repository or download the project files.
2. Set up an Anaconda environment and install required packages:
```bash
conda create -n rag_env python=3.10
conda activate rag_env
pip install pandas numpy rank_bm25 sentence-transformers openai python-dotenv
```
3. Ensure your OpenAI API key is set in the code (for local testing, you can directly assign it).

## Usage
1. Open the Jupyter Notebook (`.ipynb`) file.
2. Run all cells sequentially:
   - Load dataset
   - Setup BM25, embeddings, and cross-encoder
   - Define the `chat_with_rag()` function
   - Run the chatbot loop
3. To evaluate performance, run the evaluation cell (or separate evaluation script) to compute Precision, Recall, MRR, and nDCG.

## Evaluation Metrics
- Precision@5: Accuracy of retrieved results
- Recall@5: Coverage of relevant results
- MRR (Mean Reciprocal Rank): Rank of the first relevant document
- nDCG (Normalized Discounted Cumulative Gain): Ranking quality

Evaluation is performed using `evaluation_set.csv` containing 5 sample queries with expected relevant documents.

## Results
- The hybrid RAG pipeline retrieves relevant documents effectively, with evaluation metrics showing top performance on the sample queries.
- Cross-Encoder ensures accurate ranking of candidates before answer generation by GPT.

## Limitations
- Cross-Encoder is computationally expensive on large datasets
- CLI-based interface only (no UI)
- Small dataset size

## Future Work
- Integrate FAISS or Chroma for scalable vector retrieval
- Build a Streamlit/Flask UI for the chatbot
- Implement caching for embeddings
- Use larger or fine-tuned language models for better responses

## References
- Rank-BM25 Library
- SentenceTransformers
- HuggingFace CrossEncoder
- OpenAI GPT API
- Python, Anaconda, VS Code

