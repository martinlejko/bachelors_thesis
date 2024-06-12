# Knowledge base approach also called RAG (Retrieval-Augmented Generation)
  * Embedding & vector database
    * The pipline: Q --> Smart search (private knowledge base) --> Q + relevant data --> LLM --> A 
  * Creating pairs Q-A and then vectorising it and using simularity search. Setting up the LLM chain and propmts
  * Using embedding because it is closely related to model *accuracy*
  * This can help us as we will be fighting *LLM halucinating*
  * Knowledge graph vs vector database as it can be even better then vectors [src](https://medium.aiplanet.com/implement-rag-with-knowledge-graph-and-llama-index-6a3370e93cdd)
    * graph can support more complex and diverse queries, the downside is that the documents nned to have relationship clearly exhibited, else knowledge graph will not be able to capture it.
    * May strugle to represent complex relationships and semantic meanings (Vector databse)

# Various LLM models
  * Not using ChatGPT API as it is a third-party which is not beneficial for our private environment
    * As in our case privacy and controllability is of utmost priority
  * *OpenAI*, *FalconAI* or *PalmAI* --> where FalconAi is open-source from hugging face but from [this](https://www.linkedin.com/pulse/implementation-knowledge-base-embedding-llm-flexidigit-technologies-8hqmc) not very accurate but the PalmAI got good results, also open-source made by google

# Vector database and vectorising 
  * use of *FAISS* vector from *langchain.vectorstores*
# Other software
  * Using *langchain* for building applications with LLMs and splitting data into chunks
  * *langchain.embidding* or *OpenAIEmbedding* this is paid and *HuggingFaceInstructEmbeddings* thi is open-source so maybe usefull

# Some key questions to ask for which approach to use:
     * What is the nature of the data and its relationships?
     * Does the data primarily consist of structured or unstructured information
     * Are there complex relationshops and dependencies between entities
     * Are efficient similarity searches and recommendations necessary?
     * Is there a need for complex graph traversals and relationshop exploration
        * In sumary --> Vector db better sor similarity-based operations and graphs for capturing and alayzing complex relationships and dependencies

# What will be needed?
    * LLM -- preferably open-source
    * Embedding Model in one of the articles *thenlper/gte-large* also *LlamaIndex* was used
        * LlamaIndex is a orchestration framework that simplifies the integration of private data with public data, it uses data ingestion, indexing etc.

* Implementation with the graph will be similar: (reading pdfs and converting them into a know-graph, storing it) Q --> retrieve relevant data, context match --> LLM A
* Knowledge graphs prove to be better for avoiding model to hallucinate (stated by an article)

# What are some problems that occure?
    * Results are not accurate enough?
    * The number of parameters to tune is overwhelming
    * PDFs are a problem. If there is a messy formatting, and the way to represent it.
    * Continuesly sync the new data into the model


# What parser to use?
    * LlammaParse is better than PyPDF as it is more accurate. #TODO: implement

# What to focus on to improve RAG?
    * Improve on the data parser and the strucure. 
    * Chunk size --> trought experiments to see what works the best
    * Optimal retrieval
    * Rerank
    * 
