from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def setup_rag():
    """Setup RAG system with LangChain."""

    print("Setting up RAG system...")

    # Initialize embeddings (same model used for indexing)
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    # Load the existing ChromaDB
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    print(f"Loaded vector store with {vectorstore._collection.count()} documents")

    # Initialize Ollama LLM (you can change the model)
    llm = Ollama(
        model="llama3.2:1b",  # Change to your preferred model
        temperature=0.7
    )

    # Create a custom prompt template
    template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """

    prompt = ChatPromptTemplate.from_template(template)

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
    )

    # Create RAG chain using LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG system ready!\n")

    # Return both the chain and retriever for flexibility
    return rag_chain, retriever

if __name__ == "__main__":
    rag_chain, retriever = setup_rag()
    print("QA chain created successfully!")
    print("Import this module in other scripts to use the RAG system.")
