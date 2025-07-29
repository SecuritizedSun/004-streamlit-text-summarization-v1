import streamlit as st
import openai

def generate_summary(api_key: str, input_text: str) -> str:
    openai.api_key = api_key

    # Use direct prompt summarization via ChatGPT
    prompt = (
        "Summarize the following text in a concise, clear paragraph:\n\n"
        f"{input_text}"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert writing assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )

    return response["choices"][0]["message"]["content"].strip()

st.set_page_config(page_title="Writing Text Summarization")
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
        summary = generate_summary(openai_api_key, txt_input)
        result.append(summary)

if result:
    st.subheader("Summary:")
    st.info(result[-1])
