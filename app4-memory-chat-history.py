# Bring in deps
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = apikey

# App Framework
st.title('Youtube GPT Creator')
prompt = st.text_input('Plug in your prompt here')

# Prompt Templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'write me a youtube video title about {topic}'
)

# Script Templates
script_template = PromptTemplate(
    input_variables = ['title'],
    template = 'write me a youtube video script base on this title TITLE: {title}'
)

# Memory
memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=memory)
sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)

# Show stuff to the screen if there's a prompt
if prompt:
    response = sequential_chain({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])
    
    with st.expander('Message History'):
        st.info(memory.buffer)