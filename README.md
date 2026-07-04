# Retrieval-Augmented Generation (RAG) System using LangChain and Ollama

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain, FAISS, Sentence Transformers, and Ollama. The system retrieves relevant information from a collection of research papers and uses a Large Language Model (Llama 3.2) to generate context-aware answers.

The project was developed as part of an MSc assignment to evaluate the impact of different chunking and retrieval strategies on the quality of generated responses.

---

## Features

* Multi-document PDF ingestion
* Document chunking using configurable chunk sizes
* Semantic embeddings using Sentence Transformers
* FAISS vector database for similarity search
* Similarity Search and Max Marginal Relevance (MMR) retrieval
* Local LLM inference using Ollama (Llama 3.2)
* Automated evaluation across multiple configurations
* CSV export of experiment results and summary statistics

---

## Project Structure

```text
RAG_Project/
│
├── Data/
│   ├── rag_paper.pdf
│   ├── transformer.pdf
│   ├── bert.pdf
│   ├── llama.pdf
│   └── chain_of_thought.pdf
│
├── results/
│   ├── baseline.csv
│   ├── small_chunks.csv
│   ├── large_chunks.csv
│   ├── mmr.csv
│   ├── all_results.csv
│   └── summary.csv
│
├── rag_experiments.py
├── README.md
└── requirements.txt
```

---

## Technologies Used

* Python 3.10+
* LangChain
* Ollama
* Llama 3.2
* FAISS
* Sentence Transformers
* HuggingFace Embeddings
* Pandas
* PyPDF

---

## Document Corpus

The document corpus consists of five research papers related to Large Language Models and Retrieval-Augmented Generation:

1. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
2. Attention Is All You Need
3. BERT: Pre-training of Deep Bidirectional Transformers
4. LLaMA: Open and Efficient Foundation Language Models
5. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

These papers provide more than 50 pages of technical content for evaluating retrieval quality and response generation.

---

## Experimental Configurations

Four different configurations were evaluated:

| Configuration | Chunk Size | Overlap | Retrieval                    |
| ------------- | ---------: | ------: | ---------------------------- |
| Baseline      |       1000 |     200 | Similarity Search            |
| Small Chunks  |        500 |      50 | Similarity Search            |
| Large Chunks  |       1500 |     200 | Similarity Search            |
| MMR Retrieval |       1000 |     200 | Max Marginal Relevance (MMR) |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/rag-project.git
cd rag-project
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the environment:

**macOS/Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Download and install Ollama from:

https://ollama.com/download

Pull the required model:

```bash
ollama pull llama3.2
```

Start the Ollama server:

```bash
ollama serve
```

---

## Running the Project

Place all research paper PDFs inside the **Data** folder.

Run the project:

```bash
python rag_experiments.py
```

The program automatically:

* Loads all PDF documents
* Creates embeddings
* Builds the FAISS vector database
* Runs all four experimental configurations
* Evaluates 20 benchmark questions
* Saves the generated answers and scores as CSV files

---

## Output

The generated results are stored inside the **results** directory.

Example:

```
results/
│
├── baseline.csv
├── small_chunks.csv
├── large_chunks.csv
├── mmr.csv
├── all_results.csv
└── summary.csv
```

The summary file contains the average score for each configuration.

Example:

| Configuration | Average Score |
| ------------- | ------------: |
| MMR Retrieval |           4.8 |
| Large Chunks  |           4.7 |
| Baseline      |           4.5 |
| Small Chunks  |           4.4 |

---

## Evaluation

The system was evaluated using twenty questions divided into three categories:

* Factual Retrieval
* Multi-Hop Reasoning
* Synthesis Across Multiple Documents

Responses were manually reviewed and scored on a scale from 0 to 5 to compare the effectiveness of each retrieval strategy.

---

## Key Findings

* MMR Retrieval produced the best overall performance by retrieving more diverse and relevant document chunks.
* Large chunking preserved contextual information and performed well on synthesis tasks.
* Smaller chunks improved retrieval precision but sometimes fragmented important context.
* Retrieval quality had a greater impact on answer quality than increasing model complexity alone.

---

## Future Improvements

Potential enhancements include:

* Semantic chunking
* Cross-encoder re-ranking
* Hybrid retrieval (BM25 + Dense Retrieval)
* Metadata filtering
* ChromaDB or Pinecone integration
* Automatic evaluation using LLM-as-a-Judge

---

## Author

**Akshit Kumar**

M.Sc. Artificial Intelligence in Business

SRH University of Applied Sciences, Berlin

---

## License

This project was developed for academic purposes as part of a university assignment.
