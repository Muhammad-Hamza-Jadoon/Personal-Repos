import langchain
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_together import Together
from langchain_openai import ChatOpenAI


from langchain_together.embeddings import TogetherEmbeddings
embeddings = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
    together_api_key="24cdbdf50106e08f6ba3328ac07f97a73eb440ae36da6cdd72f9b091ccca850a"                          
)

from langchain_community.vectorstores import Chroma
# persist_directory = 'docs/chroma/file_n'
# persist_directory = 'docs/chroma/vector-kaggle'
persist_directory = 'docs/chroma/CSLectures'
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)


import chainlit as cl
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


@cl.on_chat_start
def setup_chain():

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to
    make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_chain_prompt = PromptTemplate.from_template(template)
    
    model = ChatOpenAI(
        base_url="https://api.together.xyz/v1",
        api_key="24cdbdf50106e08f6ba3328ac07f97a73eb440ae36da6cdd72f9b091ccca850a",
        model="META-LLAMA/LLAMA-3-8B-CHAT-HF",
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # llm_chain = ConversationalRetrievalChain.from_llm(
    #     model,
    #     memory=memory,
    #     retriever=vectordb.as_retriever(search_type = "mmr")
    #     # retriever=self_query_retriever,
    #     # retriever=compression_retriever,     
    # )

    llm_chain = RetrievalQA.from_chain_type(
        model,
        memory=memory,
        retriever=vectordb.as_retriever(search_type = "mmr"),
        # retriever=self_query_retriever,
        # retriever=compression_retriever,  
    
        # return_source_documents=True,
        # chain_type="map_reduce"
        # chain_type="refine"
        chain_type_kwargs={"prompt": QA_chain_prompt}    
    )
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def handle_messag(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    
    response = await llm_chain.acall(message.content, 
                                     callbacks=[
                                     cl.AsyncLangchainCallbackHandler()])

    
    # await cl.Message(response["answer"]).send()
    await cl.Message(response["result"]).send()

# chainlit run chat.py -w --port 8080
