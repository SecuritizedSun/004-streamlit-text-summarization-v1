import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

def generate_response(txt, api_key):
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo"
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce"
    )
    return chain.run(docs)

st.set_page_config(
    page_title="Writing Text Summarization"
)
st.title("Writing Text Summarization")

txt_input = st.text_area(
    "Enter your text",
    "",
    height=200
)

result = []
with st.form("summarize_form", clear_on_submit=True):
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        disabled=not txt_input
    )
    submitted = st.form_submit_button("Submit")
    if submitted and openai_api_key.startswith("sk-"):
        response = generate_response(txt_input, openai_api_key)
        result.append(response)

if len(result):
    st.info(result[-1])
