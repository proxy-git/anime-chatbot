"""Create a ChatVectorDBChain for question/answering."""
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
import pinecone
from dotenv import dotenv_values

config = dotenv_values(".env")
openai_api_key = config["OPENAI_API_KEY"]
index_name = config['PINECONE_INDEX']

pinecone.init(
        api_key=config['PINECONE_API_KEY'],  # find api key in console at app.pinecone.io
        environment=config['PINECONE_API_ENV']  # find next to api key in console
)

def get_chain(question_handler, stream_handler
) -> ConversationalRetrievalChain:
    """Create a ConversationalRetrievalChain for question/answering."""
    # Construct a ConversationalRetrievalChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    embedding = OpenAIEmbeddings(openai_api_key=config['OPENAI_API_KEY'])
    vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embedding)

    question_gen_llm = OpenAI(
        temperature=0,
        verbose=True,
        callback_manager=question_manager,openai_api_key=openai_api_key
    )
    streaming_llm = OpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,openai_api_key=openai_api_key
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain(verbose=True,retriever=vectorstore.as_retriever(),        
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
        memory=memory
    )
    return qa
