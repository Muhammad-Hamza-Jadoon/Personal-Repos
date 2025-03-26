from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_together import Together
from langchain.memory.buffer import ConversationBufferMemory
import chainlit as cl

template = """
You are a chatbot named "PotterBot" with expertise in the world of Harry potter fiction.
You are always very poetic in your response.
You do not provide information outside of this.
If a question is not about harry potter, respond with,
"I specialize only in harry potter related queries" and end your response.
donot provide any irrelevant information and end your response immediately.
Chat History: {chat_history}
keep in context the chat history aaswell
Question: {question} 
Answer:"""

prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=template
)

together_api_key="24cdbdf50106e08f6ba3328ac07f97a73eb440ae36da6cdd72f9b091ccca850a"

@cl.on_chat_start
def quey_llm():
    llm = Together(
        model="MISTRALAI/MIXTRAL-8X7B-INSTRUCT-V0.1", 
        temperature=0.7,
        together_api_key=together_api_key
    )
    conversation_memory = ConversationBufferMemory(memory_key="chat_history",
                                                   max_len=50,
                                                   return_messages=True,
                                                   )
    
    llm_chain = LLMChain(llm=llm, 
                         prompt=prompt_template,
                         memory=conversation_memory)
    
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def query_llm(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    
    response = await llm_chain.acall(message.content, 
                                     callbacks=[
                                         cl.AsyncLangchainCallbackHandler()])
    
    await cl.Message(response["text"]).send()


# !chainlit chatbot_basics.py -w --port 8080
# !chainlit run chatbot.py -w --port 8080

