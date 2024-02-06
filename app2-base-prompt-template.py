# Bring in deps
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ['OPENAI_API_KEY'] = apikey

# App Framework
st.title('Youtube GPT Creator')
prompt = st.text_input('Plug in your prompt here')

# Prompt Templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'write me a youtube video title about {topic}'
)

# Llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)

# Show stuff to the screen if there's a prompt
if prompt:
    response = title_chain.run(topic=prompt)
    st.write(response)