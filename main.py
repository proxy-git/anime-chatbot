"""Main entrypoint for the app."""
import logging
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from pydantic import BaseModel
from dotenv import dotenv_values
config = dotenv_values(".env")
openai_api_key = config["OPENAI_API_KEY"]
index_name = config['PINECONE_INDEX']

pinecone.init(
        api_key=config['PINECONE_API_KEY'],  # find api key in console at app.pinecone.io
        environment=config['PINECONE_API_ENV']  # find next to api key in console
)

class Document(BaseModel):
    name: str 
    description: str | None = None
    loader:str = "wikipedia"

app = FastAPI()
origins = [
     "http://localhost:9000",
     "http://localhost:8080",
     "http://127.0.0.1:9000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")
vectorstore = None

def upsert(document:Document):
    if not document or not document.name or not document.loader:
        return None
    if document.loader == "wikipedia":
        """Get documents from web pages."""
        # loader = ReadTheDocsLoader(path="langchain\en\latest", encoding="utf-8")
        # raw_documents = loader.load()
        raw_documents = WikipediaLoader(query=document.name, load_max_docs=2).load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        documents = text_splitter.split_documents(raw_documents)
        embedding = OpenAIEmbeddings(openai_api_key=config['OPENAI_API_KEY'])
        print(len(documents)," documents found")
    
        vectorstore = Pinecone.from_documents(documents=documents, embedding=embedding,index_name=index_name)
        print("Done")
        return document

@app.post("/upsert")
async def create_item(document: Document):
    return upsert(document)

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(question_handler, stream_handler)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            chat_history.append((question, result["answer"]))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
