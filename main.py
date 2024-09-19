from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_community.llms.ctransformers import CTransformers

import streamlit as st
from streamlit_chat import message
from utils import * 

from datetime import datetime

print('Main: Initializing............\n\n')

st.subheader("Chatbot with ChromaDB, Langchain, Llama and Streamlit")

if 'responses' not in st.session_state:
    print('Main:  Generating session stare response.......\n\n')
    
    st.session_state['responses'] = ["How can I Assist you"]

if 'requests' not in st.session_state:
    print('Main:  Generating session state request.......\n\n')
    st.session_state['requests'] = []

start1 = datetime.now()
llm = CTransformers(model = r'Model\llama-2-7b-chat.ggmlv3.q4_0.bin',
                        model_type = 'llama',
                        config={'max_new_tokens': 100,
                              'temperature': 0.01,
                              'context_length': 1600})

print(f'Main:  LLM loaded in {(datetime.now() - start1).total_seconds()}s.......\n\n')

if type(llm) == tuple:
    llm = llm[0]
    print(f'Main:  LLM  was a tuple...............\n\n')

if 'buffer_memory' not in st.session_state:
    print('Main:  Creating Buffer Memory in session.......\n\n')
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages = True)

template = """Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'"""
sys_msg_template = SystemMessagePromptTemplate.from_template(template=template)

human_msg_template = HumanMessagePromptTemplate.from_template(template= "{input}")

prompt_template = ChatPromptTemplate.from_messages([sys_msg_template, MessagesPlaceholder(variable_name = 'history'),
                                                    human_msg_template])

start3 = datetime.now()

conversation = ConversationChain(memory= st.session_state.buffer_memory, prompt= prompt_template, llm= llm, verbose= True)
print(f'Main:  writing conversation from session Buffer Memory in {(datetime.now()-start3).total_seconds()}s......\n\n')
print(f'{conversation =     }\n\n')

# Container for Chat History
print('Main:  Generating response containter.......\n\n')
response_container = st.container()

# Container for Text Box
print('Main:  Generating text container.......\n\n')
text_container = st.container()


with text_container:
    query = st.text_input("Query", key= "input")
    print('Main:  Asking for user Query.......\n\n')

    if query:
        with st.spinner("typinggg........."):
            print('Main:  Got a Query.......\n\n')
            conversation_string = get_conversation_string()
            print('Main:  Got Conversation String .......\n\n')

            st.code(conversation_string)
            print('Main:  writing conversation string code.......\n\n')

            refined_query = query_refiner(conversation= conversation_string, query= query)
            print('Main:  Got refined Query.......\n\n')

            st.subheader("Refined Query:")
            print('Main:  writing Refined Query.......\n\n')

            st.write(refined_query)

            context = find_match(input= refined_query)
            print('Main:  Context match found.......\n\n')
            print("context:   ",context, "\n\n")

            start2 = datetime.now()

            response = conversation.predict(input = f"Context:\n {context} \n\n Query:\n {query}")
            print(f'Main:  Conversation predicted response in {(datetime.now() - start2).total_seconds()}s.......\n\n')

        st.session_state.requests.append(query)
        print('Main:  appending query to session state requests......\n\n')

        st.session_state.responses.append(response)
        print('Main:  appending response to session state reasponses......\n\n')


with response_container:
    print('Main:  In response container......\n\n')
    if st.session_state.responses:

        print('Main:  Got session state response......\n\n')
        print(f'Main:  {len(st.session_state.responses) =   } and {len(st.session_state.requests) =   }..\n\n')

        for i in range(len(st.session_state.responses)):
            message(st.session_state['responses'][i], key= str(i))
            print(f'Main:  Wrote a response message for {i = }......\n\n')
            print('=x='*24)

            if i < len(st.session_state['requests']):
                message(st.session_state['requests'][i], is_user= True, key= str(i) + "_user")
                print(f'Main:  Wrote a request message for {i = }......\n\n')
                print('=|='*24)
        