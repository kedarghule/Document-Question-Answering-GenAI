# Document Question Answering using LLMs

This project builds an document question answering app powered by Large Language Models (LLMs) like Falcon-7B and Dolly-v2-3B using LangChain, the ChromaDB vector database. It is deployed on Streamlit.

Link to app: https://document-question-answering-kedarghule.streamlit.app/

Note: Due to memory issues with Streamlit, the app may not work sometimes and give an error. This is due to the 1GB memory limit by Streamlit. Here is a video that shows how the app works: https://drive.google.com/file/d/1nkvdqdx1eMWTZqhkyzU_2IJZgOg-uS8O/view?usp=sharing


https://github.com/kedarghule/Document-Question-Answering/assets/41315903/70b46f0d-f2ac-473b-aa25-a99afef8e5ec



## Problem Statement

In today's era of information overload, individuals and organizations are faced with the challenge of efficiently extracting relevant information from vast amounts of textual data. Traditional search engines often fall short in providing precise and context-aware answers to specific questions posed by users. As a result, there is a growing need for advanced natural language processing (NLP) techniques to enable accurate document question answering (DQA) systems.

The goal of this project is to develop a Document Question Answering app powered by Large Language Models (LLMs), such as Falcon-7B and Dolly-v2-3B, utilizing the LangChain platform and the ChromaDB vector database. By leveraging the capabilities of LLMs, this app aims to provide users with accurate and comprehensive answers to their questions within a given document corpus.

## Methodology

- **Document Loading**: The app supports uploading of `.txt` files and `.docx` files. Once uploaded, the `.docx` file is converted to a `.txt` file. Using LangChain, the document is loaded using TextLoader.
- **Text Splitting**: Next, we split the text recursively by character. For this, we use the RecursiveCharacterTextSplitter. This text splitter is the recommended one for generic text. A chunk size is specified as well as a list of separators.
- **Generating Embeddings**: The HuggingFaceEmbeddings() class in LangChain uses the sentence_transformers embedding models, more specifically the mpnet-base-v2 model.
- **Vector Database**: A vector store using ChromaDB is used to store embedded data and to perform vector search operations.
- **Context Search**: Depending on the user's question, a similarity search is carried out on the vector database to get the most appropriate context to answer the question.
- **Prompt Engineering**: The above context, along with an appropiate prompt and the user's question is created.
- **Inference using LLMs**: Depending on the user's choice, we either use Falcon-7B or Dolly-v2-3B for our inference. We pass the engineered prompt to the model ad display the model's response.
