import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.ctransformers import CTransformers
import chromadb
from langchain_community.vectorstores import Chroma
from datetime import datetime


start1 = datetime.now()

llm_utils = CTransformers(model = r'Model\llama-2-7b-chat.ggmlv3.q4_0.bin',
                        model_type = 'llama',
                        config={'max_new_tokens': 100,
                              'temperature': 0.01,
                              'context_length': 1600})
print(f'Utils:  Defined LLM in {(datetime.now() - start1).total_seconds()}s ......\n\n')

if type(llm_utils) == tuple:
    llm_utils = llm_utils[0]
    print('Utils:  LLM  was a tuple...............\n\n')

start5 = datetime.now()

persistent_client = chromadb.PersistentClient(path = "./chroma_langchain_medinfo_db")
vector_store_from_client = Chroma(
    client= persistent_client,
    embedding_function=HuggingFaceEmbeddings()
)
print(f'Utils:  Persistant client and vector store client loaded in {(datetime.now()-start5).total_seconds()}s.......\n\n')


def find_match(input: str):
    print(f'Utils:  Find match {input =     }.......\n\n')
    start2 = datetime.now() 

    results = vector_store_from_client.similarity_search(
        input ,
        k=2
    )
    print(f'Utils:  found results in {(datetime.now() - start2).total_seconds()}s......\n\n')
    return results[0].page_content + "\n" + results[1].page_content


def query_refiner(conversation, query):
    print("Utils:         query refiner is called.............\n\n")
    print(f"{conversation =     }")
    print(f"{query =     }")
    print('\n\n')
    start3 = datetime.now()

    template = """ Given the following user query and conversion log, formulate a question that would be the most
                relevant to provide the user with  an answer from a knowledge base.
                
                Conversation Log : 
                {conversation} 
                
                Query :
                {query}
                
    Refined Query: """
    
    prompt = PromptTemplate(input_variables= ['conversation','query'], template= template)

    response = llm_utils.invoke(prompt.format(conversation=conversation, query=query))
    print(f'Utils:  query refiner response generated in {(datetime.now() - start3).total_seconds()}s......\n\n')
    print('query refiner response:\n',response)
    print('\n\n')

    return response 


def get_conversation_string():
    print('Utils:    get_conversation_string is called  \n\n')
    print(f'Utils:    {len(st.session_state['responses']) =        }')

    start4 = datetime.now()

    conversation_string = ""

    j = 0
    for i in range(len(st.session_state['responses']) - 1):

        conversation_string += "Human: " + st.session_state['requests'] [i] + "\n"
        print(f"{j}th conv:           {conversation_string}")

        conversation_string += "Bot: " + st.session_state['responses'] [i+1] + "\n"
        print(f"{j}th conv:           {conversation_string}")
        j += 1
    print(f'Utils:  get_conversation_string executed in {(datetime.now() - start4).total_seconds()}s......\n\n')

    return conversation_string
