import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# Step 1: Load documents
def load_documents(folder):
    texts = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            try:
                with open(os.path.join(folder, file), "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
            except Exception as e:
                print(f"Error reading {file}: {e}")
    return texts

# Step 2: Split into chunks
def split_documents(docs):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents(docs)

# Step 3: Embed and store in FAISS
def embed_and_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    return db

# Step 4: Ask questions
def ask_question(db, question, qa_model):
    docs = db.similarity_search(question, k=2)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return qa_model(prompt)[0]['generated_text']

# Main CLI function
def main():
    print("📄 Loading and processing documents...")
    raw_docs = load_documents("documents")
    chunks = split_documents(raw_docs)
    db = embed_and_store(chunks)

    print("🤖 Loading tiny model for QA...")
    qa_model = pipeline("text2text-generation", model="google/flan-t5-small")

    print("🔎 Ask your question (type 'exit' to quit):")
    while True:
        query = input(">>> ")
        if query.lower() in ['exit', 'quit']:
            break
        response = ask_question(db, query, qa_model)
        print(f"\nAnswer:\n{response}\n")

if __name__ == "__main__":
    main()
