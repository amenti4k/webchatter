# $streamlit run main.py
# pip install streamlit langchain langchain_openai beautifulsoup4 python-dotenv chromadb

# God's gift to datascientists like me that love products too (product ds > research ds)
import streamlit as st
import pandas as pd
import numpy as np
# for separating out the chat that happens between AIMessage and HumanMessage (makes it easier to read))
from langchain_core.messages import AIMessage, HumanMessage

# below is where we get the chunks of the website scapped with beautifulsoup4
from langchain_community.document_loaders import WebBaseLoader

# to chunk the text into pieces 
from langchain.text_splitter import RecursiveCharacterTextSplitter

# to get the embeddings of the chunks so get chroma from the community packages
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

#get the context chat prompt template
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


import os 

my_secret = os.environ['OPENAI_API_KEY'] 

def get_response(user_input):
    # this should be backend and done once and for all 
    retriever_chain = get_context_retriver_chain(st.session_state.vector_store) #This part just runs once and then we can just use the retriever chain to get #the response
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain) ##then ask the question and get the response
    #return get_conversational_rag_chain
    response = conversation_rag_chain.invoke({
        "chat_history":st.session_state.chat_history, 
        "input": user_query
    })
    return response ['answer'] #leave out the other bullshit I don't need

# Getting the website vectorized
# def get_vectorstore_from_url(url):
# get the entire vector store
def get_vectorstore_from_url(url):
    # get the text in docuemnt form to enable debugging 
    loader = WebBaseLoader(url)
    document = loader.load()

    #use the langchain splitter using the chunks 
    text_splitter = RecursiveCharacterTextSplitter()#chunk_size=1000, chunk_overlap=0)
    document_chunks = text_splitter.split_documents(document)

    #create vector store from the chunks (where are we storing and where are we getting the embeddings)
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store

def get_context_retriver_chain(vector_store):
    # initalize a language model 
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        #add the chat history into the new prompt by having the messages placeholder variable 
        MessagesPlaceholder(variable_name = "chat_history"), 
        #when you populate the prompt, the input changes
        ("user", "{input}"),
        ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation") #this is another prompt that enables the combination of #the new imput + the history 
    ])

    #uses the LLM + the vector + the prompt to create a chain
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    # answer the question based on what we have 
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)

    # now link to the retriever chain 
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# page config
st.set_page_config(page_title="Amenti's Website Chatter", page_icon = "🧿") 
st.title("Yo! Paste in live websites, and chat with them.") 
website_url = st.text_input("Website URL")

# we're going to initalize a list for the chat history

# sidebar
df = pd.DataFrame({
    "col1": np.random.randn(1000) / 50 + 37.76,
    "col2": np.random.randn(1000) / 50 + -122.4,
    "col3": np.random.randn(1000) * 100,
    "col4": np.random.rand(1000, 4).tolist(),
})

with st.sidebar: 
    st.write("a simple front to navigate the web faster and do your own deep dives and keep asking more questions. new ways of 'surfing' the web.  X...Amenti") 

# Don't open the chatbox unless someone enters the website url
if website_url is None or website_url == "":
    st.error("Please enter a website URL (make sure it's the https link)")

else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]

    # this part is to just make sure we're not running our money dry 
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # user input and history handling
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "": 
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
