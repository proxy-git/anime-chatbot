# Anime Chatbot

This repo is an implementation of a locally hosted chatbot specifically focused on question answering documents from Wikipedia.
Built with [LangChain](https://github.com/hwchase17/langchain/) and [FastAPI](https://fastapi.tiangolo.com/).

The app leverages LangChain's streaming support and async API to update the page in real time for multiple users.

## Running locally
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `make start` or the start code in the Makefile
3. Open [localhost:9000](http://localhost:9000) in your browser.
4. Enter the Wikipedia Page title in the "Enter wiki page to add" input field and click "Add". The data will be added to your Pinecone DB
5. Enter your queries in the "Write your question" input field