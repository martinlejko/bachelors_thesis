import os
from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


def setup_environment() -> None:
    """Set up environment variables and configurations."""
    os.environ["USER_AGENT"] = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"
    )


def load_data(urls: List[str]) -> List[Document]:
    loader = WebBaseLoader(urls)
    return loader.load()


def process_data(
    documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 0
) -> List[Document]:
    """Split documents into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def create_vectorstore(
    documents: List[Document], model_name: str = "nomic-embed-text"
) -> Chroma:
    """Create and return a vector store from processed documents."""
    try:
        embeddings = OllamaEmbeddings(model=model_name)
        return Chroma.from_documents(documents=documents, embedding=embeddings)
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        raise


def setup_model(model_name: str = "llama3.1:8b") -> ChatOllama:
    """Initialize and return the language model."""
    return ChatOllama(model=model_name)


def create_prompt_template() -> ChatPromptTemplate:
    """Create and return the RAG prompt template."""
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
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_qa_chain(vectorstore: Chroma, model: ChatOllama, prompt: ChatPromptTemplate):
    """Create the question-answering chain."""
    retriever = vectorstore.as_retriever()
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )


def main():
    # Example URLs
    urls = [
        "https://d3s.mff.cuni.cz/teaching/nswi200/teams/",
        "https://d3s.mff.cuni.cz/teaching/nprg035/",
        "https://webik.ms.mff.cuni.cz/nswi142/",
        "https://martinlejko.github.io/posts/hello-blog/",
    ]

    # Setup and initialization
    setup_environment()

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
    response = qa_chain.invoke(question)
    print(response)


if __name__ == "__main__":
    main()
