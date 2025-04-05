## Main Functions

The iterations folder consists of users iterations where he utilizes already existing code to try out new ideas. To see if the evaluation of the RAG pipeline can be improved.:

**Iteration 0**: Uses more robust code from the src folder and adds cashing, different sources and a more robust evaluation pipeline.
**Iteration 1**: Utilizes the ntlk library to radically clean the data, with methods like deleting the stop words or lematization.
**Iteration 2**: In iteration 3, I have tried to step back and try the minimalistic approach of cleaning the data. That is lowercasing it and removing white spaces.
**Iteration 3**: In iteration 4, I have tried moderate approach of cleaning the data. By removing special characters, lowercasing the data and also normalizing it to unicode.
**Iteration 4**: In iteration 5, I have tried to use the more advanced approach of cleaning the data. By removing special characters, lowercasing the data and using regex to remove headers and footers.
**Iteration 5**: In iteration 6, I have tried to use a different prompt template. The new prompt template is more specific, gives the LLM more context, specialization and provides more rules.
**Iteration 6**: In iteration 7, I have tried to use a different chunking method. The default chunking size is 500 characters, so I have decided to try 1000 characters with 10% overlap. This gives the LLM more context and allows it to generate better answers.
**Iteration 7**: In iteration 8, I have tried to use a different embedding model. We have used a bigger and therefor stronger model. To see if it improves the results.
**Iteration 8**: In iteration 9, I have tried to edit the vector retrieval parameters. The default retrieval parameters are 5, so I have decided to up that to 8. So the LLM has more context.
**Iteration 9**: In iteration 10, I have tried to use a different retrieval method. That is the mmr method.