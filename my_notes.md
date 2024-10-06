# Embedding
  * Type of vector, (big,small,animal tree) compass
  * Turning data into vector in hundres of dimensions by some software
  * After turning it into Vector embedding --> Vector database

# YT video: Knowledge embedding
  * Case study --> generating the customer replies with companies best practice
  * [youtube vid](https://www.youtube.com/watch?v=c_nCjlSB1Zk&list=PLd6t9g4NfNFrM69CYx56NzwG1gbOAw3uO)
  * Using a *text-splitter* for big chunk of data, that we will have
  * For embidding he uses *OpenAIEmbeddings* and also *FAISS*
  * Use of a third-party website for sharing

# Medium series about LLM
  * [medium](https://cismography.medium.com/knowledge-bases-and-retrieval-augmented-llms-a-primer-c054db532b91)
  
  * What is *langchain*?
    * Open-source framework that makes it easier to build applications LLM
  * Using *langchain* with [gpt4all](https://gpt4all.io/index.html)
  * use of *langchain* as the embedder here [embedding LLM with langchain](https://www.linkedin.com/pulse/implementation-knowledge-base-embedding-llm-flexidigit-technologies-8hqmc)

* How does vector databse work?
  * Each entity is represented as a high-dimensional vector and the similarity between entities is calculated based on vecotr distance

* Nice source of material and frameworks: [llamahub](https://llamahub.ai/)
* Plaban Nayak [k graph vs vector db](https://medium.aiplanet.com/implement-rag-with-knowledge-graph-and-llama-index-6a3370e93cdd) tried to connect with him


# Benchmarking LLMs
  * Types of benchmarks and what do they test [benchmarks](https://symbl.ai/developers/blog/an-in-depth-guide-to-benchmarking-llms/)
  * Important to us are HellaSwag as it tests commonsense reasoning and natural language inference, (maybe not as ti doesnt test reasoning for specialised domains)
  * MMLU tests how well it understands the language and subsequently its ability to solve problems with the knowledge
  * TruthfulQA tests the benchmark for hallucination 
  * MT Bench --> tests longer conversations about the same topic and how well it can follow

# LLM Leaderboards
  * we can use [HuggingFaceLeaderboard](https://huggingface.co/collections/open-llm-leaderboard/the-big-benchmarks-collection-64faca6335a7fc7d4ffe974a) which test these LLMS on the benchmarks above 
  * [arxiv mistral 7B](https://arxiv.org/abs/2310.06825) Text about how mistral 7B outperforms llama2 13B in every benchmark
  * Another website for tierlists [site](https://llm.extractum.io/)
  
