from dotenv import load_dotenv 

from langchain_groq import ChatGroq 
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage,SystemMessage 
from langchain_community.utilities import SerpAPIWrapper 
from langgraph.checkpoint.memory import InMemorySaver
import os 
import streamlit as st 


load_dotenv()
api_key=os.getenv("GROQ_API_KEY")
serpapi_api_key=os.getenv("SERP_API_KEY")
#memory 
if "checkpoint" not in st.session_state:
    st.session_state.checkpoint = InMemorySaver()

checkpoint = st.session_state.checkpoint


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
    search=SerpAPIWrapper(
        serpapi_api_key=serpapi_api_key
    )
    agent=create_agent(
        model=llm,
        tools=[search.run],
        system_prompt="you are helpful assistant",
        checkpointer=checkpoint
    )
    with st.spinner("thinking....."):
         response=agent.invoke({
             "messages":[
                 HumanMessage(content=user_input)
             ]
         },
         config={
    "configurable": {
        "thread_id": "user_1"
    }
}
       
       
    )

    st.divider()
    st.subheader("answer")
    st.write(response["messages"][-1].content)


        



