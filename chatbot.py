from dotenv import load_dotenv 

from langchain_groq import ChatGroq 
from langchain_core.messages import HumanMessage,SystemMessage 
import os 
import streamlit as st 


load_dotenv()
api_key=os.getenv("GROQ_API_KEY")

st.title("GROQ CHAT BOT")
st.caption("ASK ANY QUESTION")
st.divider()

with st.form("Chat Form"):
    user_input=st.text_area("your message",placeholder="type your question here ")
    submitted=st.form_submit_button("send") 
if submitted:
    llm=ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.5,
        max_retries=2,
        max_tokens=None,   
        timeout=None,
        reasoning_format="parsed",
        api_key=api_key
    )
    with st.spinner("thinking....."):
         response=llm.invoke([
        SystemMessage(content="your are helpful assistant"),
        HumanMessage(content=user_input)
    ])

    st.divider()
    st.subheader("answer")
    st.write(response.content)


        



