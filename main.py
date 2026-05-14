import streamlit as st
from RAG_Helper import get_qa_chain, create_vector_db

st.title("FAQ Bot")
btn = st.button("Refresh Data")
if btn:
    create_vector_db()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])