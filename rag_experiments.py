
import os
import json
import pandas as pd
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

# =========================
# CONFIG
# =========================

DATA_FOLDER = "Data"   # Put at least 5 PDFs here
MODEL_NAME = "llama3.2"

QUESTIONS = [

# FACTUAL RETRIEVAL (1-8)

"What is Retrieval-Augmented Generation (RAG)?",

"Who introduced the Transformer architecture?",

"What is the main contribution of the BERT paper?",

"What does LLaMA stand for?",

"What is Chain-of-Thought prompting?",

"Which paper introduced self-attention as the primary sequence modeling mechanism?",

"What benchmark datasets were used in the BERT paper?",

"What problem is RAG designed to solve?",


# MULTI-HOP REASONING (9-14)

"How does RAG improve upon the limitations of standard language models?",

"Compare the Transformer architecture with BERT.",

"How does LLaMA differ from BERT in terms of training objectives?",

"What role does attention play in both Transformer and LLaMA models?",

"How do Chain-of-Thought prompting and RAG improve reasoning in different ways?",

"What common challenges are addressed by BERT and LLaMA?",


# SYNTHESIS QUESTIONS (15-20)

"Which of the five papers had the greatest impact on modern large language models and why?",

"Explain how RAG, BERT, and LLaMA could be combined in a production AI system.",

"What are the major trade-offs between retrieval-based and parametric knowledge systems?",

"Summarize the evolution from Transformers to modern LLMs using all five papers.",

"What future research directions are suggested across these papers?",

"Provide a comprehensive synthesis of all five papers and explain how they collectively contributed to today's AI systems."

]

CONFIGURATIONS = [
    {
        "name": "baseline",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "retrieval": "similarity"
    },
    {
        "name": "small_chunks",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "retrieval": "similarity"
    },
    {
        "name": "large_chunks",
        "chunk_size": 1500,
        "chunk_overlap": 200,
        "retrieval": "similarity"
    },
    {
        "name": "mmr",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "retrieval": "mmr"
    }
]

# =========================
# LOAD DOCUMENTS
# =========================

def load_documents():
    docs = []

    pdfs = list(Path(DATA_FOLDER).glob("*.pdf"))

    if len(pdfs) < 5:
        print("WARNING: Assignment requires at least 5 documents.")

    for pdf in pdfs:
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()

        for page in pages:
            page.metadata["source_file"] = pdf.name

        docs.extend(pages)

    print(f"Loaded {len(docs)} pages from {len(pdfs)} PDFs")
    return docs


# =========================
# BUILD VECTOR STORE
# =========================

def build_vector_store(documents, chunk_size, chunk_overlap):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    print(f"Chunks Created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_db, chunks


# =========================
# RETRIEVE
# =========================

def retrieve_docs(vector_db, query, retrieval_type):

    if retrieval_type == "mmr":
        retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 20
            }
        )
    else:
        retriever = vector_db.as_retriever(
            search_kwargs={"k": 3}
        )

    return retriever.invoke(query)


# =========================
# GENERATE ANSWER
# =========================

def answer_question(llm, vector_db, question, retrieval_type):

    docs = retrieve_docs(
        vector_db,
        question,
        retrieval_type
    )

    context = "\n\n".join(
        [d.page_content for d in docs]
    )

    prompt = f"""
Answer ONLY from the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    answer = llm.invoke(prompt)

    return answer


# =========================
# SIMPLE SCORING
# =========================

def score_answer(answer):

    if len(answer.strip()) < 20:
        return 1

    if len(answer.strip()) < 100:
        return 3

    return 5


# =========================
# EVALUATE CONFIG
# =========================

def evaluate_configuration(config, docs, llm):

    print("\\n" + "=" * 80)
    print(f"RUNNING: {config['name']}")
    print("=" * 80)

    vector_db, chunks = build_vector_store(
        docs,
        config["chunk_size"],
        config["chunk_overlap"]
    )

    rows = []

    for idx, question in enumerate(QUESTIONS, start=1):

        print(f"Question {idx}/20")

        answer = answer_question(
            llm,
            vector_db,
            question,
            config["retrieval"]
        )

        score = score_answer(answer)

        rows.append({
            "configuration": config["name"],
            "question_id": idx,
            "question": question,
            "answer": answer,
            "score": score,
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "retrieval": config["retrieval"]
        })

    return pd.DataFrame(rows)


# =========================
# MAIN
# =========================

def main():

    os.makedirs("results", exist_ok=True)

    llm = OllamaLLM(model=MODEL_NAME)

    docs = load_documents()

    all_results = []

    for config in CONFIGURATIONS:

        df = evaluate_configuration(
            config,
            docs,
            llm
        )

        df.to_csv(
            f"results/{config['name']}.csv",
            index=False
        )

        all_results.append(df)

    combined = pd.concat(all_results)

    combined.to_csv(
        "results/all_results.csv",
        index=False
    )

    summary = (
        combined
        .groupby("configuration")["score"]
        .mean()
        .reset_index()
        .rename(columns={
            "score": "average_score"
        })
    )

    summary.to_csv(
        "results/summary.csv",
        index=False
    )

    print("\\nSUMMARY")
    print(summary)

    print("\\nFinished Successfully!")


if __name__ == "__main__":
    main()
