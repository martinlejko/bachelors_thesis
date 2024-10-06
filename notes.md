# Knowledge base approach
  * Embedding & vector database
    * The pipline: Q --> Smart search (private knowledge base) --> Q + relevant data --> LLM --> A 
  * Creating pairs Q-A and then vectorising it and using simularity search. Setting up the LLM chain and propmts
  * Using embedding because it is closely related to model *accuracy*


# Various LLM models
  * Not using ChatGPT API as it is a third-party which is not beneficial for our private environment
    * As in our case privacy and controllability is of utmost priority
  * *OpenAI*, *FalconAI* or *PalmAI* --> where FalconAi is open-source from hugging face but from this not very accurate but the PalmAI got good result/

# Vector database and vectorising 
  * use of *FAISS* vector from *langchain.vectorstores*
# Other software
  * Using *langchain* for building applications with LLMs and splitting data into chunks
  * *langchain.embidding* or *OpenAIEmbedding* this is paid and *HuggingFaceInstructEmbeddings* thi is open-source so maybe usefull
