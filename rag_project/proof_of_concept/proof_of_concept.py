"""
Retrieval-Augmented Generation (RAG) Proof of Concept

This module implements a basic RAG system using LangChain components and Ollama models.
The system loads data from web URLs, processes it into chunks, embeds them into a vector store,
and creates a question-answering chain that retrieves relevant context to answer user queries.
"""

from typing import Any, List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


def load_data(urls: List[str]) -> List[Document]:
    """
    Load data from a list of web URLs into Document objects.

    Args:
        urls: List of URLs to fetch data from

    Returns:
        List of Document objects containing the loaded content
    """
    loader = WebBaseLoader(urls)
    return loader.load()


def process_data(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 0) -> List[Document]:
    """
    Split documents into chunks for processing.

    Args:
        documents: List of Document objects to process
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks

    Returns:
        List of Document objects after splitting
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)


def create_vectorstore(documents: List[Document], model_name: str = "nomic-embed-text") -> Chroma:
    """
    Create and return a vector store from processed documents.

    Args:
        documents: List of Document objects to embed
        model_name: Name of the embedding model to use

    Returns:
        Chroma vector store containing embeddings

    Raises:
        Exception: If vectorstore creation fails
    """
    try:
        embeddings = OllamaEmbeddings(model=model_name)
        return Chroma.from_documents(documents=documents, embedding=embeddings)
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        raise


def setup_model(model_name: str = "llama3.1:8b") -> ChatOllama:
    """
    Initialize and return the language model.

    Args:
        model_name: Name of the Ollama model to use

    Returns:
        Configured ChatOllama model instance
    """
    return ChatOllama(model=model_name)


def create_prompt_template() -> ChatPromptTemplate:
    """
    Create and return the RAG prompt template.

    Returns:
        ChatPromptTemplate configured for RAG question answering
    """
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved 
    context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    <context>
    {context}
    </context>
    Answer the following question:
    {question}
    """
    return ChatPromptTemplate.from_template(template)


def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents into a single string.

    Args:
        docs: List of Document objects to format

    Returns:
        String containing all document contents joined by newlines
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_qa_chain(vectorstore: Chroma, model: ChatOllama, prompt: ChatPromptTemplate):
    """
    Create the question-answering chain that returns both answer and retrieved documents.

    Args:
        vectorstore: Vector store with embedded documents
        model: Language model for answering questions
        prompt: Prompt template for the QA task

    Returns:
        Function that takes a question and returns answer with retrieval context
    """
    retriever = vectorstore.as_retriever()

    # Create the standard QA chain
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | model | StrOutputParser()
    )

    def invoke_with_retrieval_context(question):
        """
        Invoke the QA chain and return both the answer and retrieval context.

        Args:
            question: The question to answer

        Returns:
            Dictionary containing the answer and retrieval context
        """
        retrieved_docs = retriever.invoke(question)
        answer = qa_chain.invoke(question)

        context_strings = [doc.page_content for doc in retrieved_docs]
        return {"answer": answer, "retrieval_context": context_strings}

    return invoke_with_retrieval_context


def print_formated_response(question: str, response: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("-" * 80)
    print(f"ANSWER: {response['answer']}")
    print("-" * 80)
    print("SOURCES:")
    for i, source in enumerate(response["retrieval_context"], 1):
        formatted_source = source.replace("\n", "")
        print(
            f"\n[Source {i}]:\n{formatted_source[:300]}..."
            if len(formatted_source) > 300
            else f"\n[Source {i}]:\n{formatted_source}"
        )
    print("=" * 80 + "\n")


def main():
    """
    Main function to demonstrate RAG pipeline functionality with example data.
    """
    # Example URLs
    urls = [
        "https://d3s.mff.cuni.cz/teaching/nswi200/teams/",
        "https://d3s.mff.cuni.cz/teaching/nprg035/",
        "https://webik.ms.mff.cuni.cz/nswi142/",
        "https://martinlejko.github.io/posts/hello-blog/",
    ]

    # Data pipeline
    raw_data = load_data(urls)
    processed_data = process_data(raw_data)
    vectorstore = create_vectorstore(processed_data)

    # Model and prompt setup
    model = setup_model()
    prompt = create_prompt_template()

    # Create QA chain
    qa_chain = create_qa_chain(vectorstore, model, prompt)

    # Example question
    question = "What is the amount of people in a team for operating systems course at MFF cuni?"
    response = qa_chain(question)

    print_formated_response(question, response)


if __name__ == "__main__":
    main()
