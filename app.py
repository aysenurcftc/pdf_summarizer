import asyncio
from documents import read_pdf, split_text
from graph import create_graph
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


st.set_page_config(page_title="PDF Summarizer", layout="centered")
st.title("PDF Summarizer with LangChain")



async def summarize_pdf(pdf_file):

    docs = read_pdf(pdf_file)
    split_docs = split_text(docs)


    graph = create_graph()
    app = graph.compile()

    summary = ""

    index = len("'final_summary':")

    async for step in app.astream(
        {"contents": [doc.page_content for doc in split_docs]},
        {"recursion_limit": 10},
    ):

        if "summary" in step:
            summary = step["summary"]
        else:
            summary = "\n".join([str(value) for key, value in step.items() if "summary" in key.lower()])

    return summary[index+3:-2]



uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])


if uploaded_file is not None:

    if st.button("Summarize"):
        with st.spinner("Summarizing the PDF..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            summary = loop.run_until_complete(summarize_pdf(uploaded_file))

        st.subheader("Summary:")
        st.write(summary)


