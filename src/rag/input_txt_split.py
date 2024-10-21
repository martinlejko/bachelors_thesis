import os

# Set a user_agent to avoid being blocked by the website
os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"
)

# Load the document from the web
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

urls = [
    "https://d3s.mff.cuni.cz/teaching/nswi200/teams/",
    "https://d3s.mff.cuni.cz/teaching/nprg035/"
]

loader = WebBaseLoader(urls)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# create a vectorstore from the documents
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

try:
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)
except Exception as e:
    print(e)

# search for similar documents and test if the vectorstore is working
question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
len(docs)

# setting up the model
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.1:8b",
)

# first approach
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

retriever = vectorstore.as_retriever()

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

# question = "What is the amout of people in a team for operating systems course at MFF cuni?"
# question = "How many points do I need to get to pass the course of c# programming?"
# question = "What is the bare minumum that i need to reach to get a credit from the course of c# programming?"
question = "How many points do I need from homeworks to get a credit from the course of c# programming?"

response = qa_chain.invoke(question)
print(response)
