# Mini RAG CLI (Retrieval-Augmented Generation)

This is a lightweight, fast, and local Retrieval-Augmented Generation (RAG) system built using [LangChain](https://python.langchain.com), FAISS, and a tiny LLM (`google/flan-t5-small`). Perfect for running document-based question-answering tasks right from your terminal.

---

## Features

- Loads all `.txt` documents from the `documents/` folder  
- Splits them into manageable chunks  
- Embeds them using `sentence-transformers/all-MiniLM-L6-v2`  
- Stores in a local FAISS vector store  
- Uses `google/flan-t5-small` for answering your queries  
- Works fully offline (after initial model download)  
- CLI-based, minimal setup, no Hugging Face login required  

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mini-rag-cli.git
cd mini-rag-cli

### 2. Install requirements

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, you can create one with the following contents:

```txt
transformers
torch
langchain
faiss-cpu
sentence-transformers
```

### 3. Add your documents

Put your `.txt` files inside a folder named `documents/` in the same directory as `rag.py`.

---

## Run the QA CLI

```bash
python rag.py
```

You’ll see:

```
Loading and processing documents...
Loading tiny model for QA...
Ask your question (type 'exit' to quit):
>>>
```

Start asking questions based on your documents!

---

## Project Structure

```
mini-rag-cli/
├── rag.py               # Main CLI script
├── documents/           # Your input text files
└── README.md            # Project documentation
```

---

## Model Details

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (fast and small, good for CPU)
- **LLM for generation**: `google/flan-t5-small` via Hugging Face Transformers pipeline

---

## Contributing

PRs and issues are welcome! Feel free to contribute better prompting strategies, faster LLMs, or UI improvements.

---


