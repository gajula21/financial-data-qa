import streamlit as st
import os
import io
import re
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import Pinecone as PineconeStore
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException

TEMP_PDF_DIR = "temp_pdfs"
os.makedirs(TEMP_PDF_DIR, exist_ok=True)

def cleanup_temp_pdfs():
    for f_name in os.listdir(TEMP_PDF_DIR):
        if f_name.endswith(".pdf"):
            try:
                os.remove(os.path.join(TEMP_PDF_DIR, f_name))
            except OSError:
                pass

@st.cache_data(show_spinner=False)
def read_uploaded_files(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Document]:
    documents = []
    for uploaded_file in files:
        file_path = os.path.join(TEMP_PDF_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(file_path)
        local_pages = loader.load()
        for p in local_pages:
            p.metadata['source'] = uploaded_file.name
        documents.extend(local_pages)
        try:
            pdf_reader = PdfReader(file_path)
            with st.expander(f"📄 Preview: {uploaded_file.name}"):
                for i, page in enumerate(pdf_reader.pages[:3]):
                    st.markdown(f"**Page {i+1}:**")
                    st.caption(page.extract_text()[:1000])
        except Exception as e:
            st.warning(f"Could not generate preview for {uploaded_file.name}: {e}")
    return documents

@st.cache_resource(show_spinner=False)
def create_qa_system(api_key: str, pinecone_key: str, _documents: List[Document], index_name: str):
    import math

    def batched(iterable, n=100):
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    if not _documents:
        return None, "No documents were provided to index."

    try:
        pc = Pinecone(api_key=pinecone_key)
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)

        index = pc.Index(index_name)
        try:
            index.delete(delete_all=True, namespace="default")
        except NotFoundException:
            pass

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(_documents)
        chunks = [c for c in chunks if c.page_content.strip()]

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        for batch in batched(chunks, 100):
            PineconeStore.from_documents(
                documents=batch,
                embedding=embeddings,
                index_name=index_name,
                namespace="default"
            )

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=api_key)

        prompt_template = """
        You are an expert financial analyst assistant. Use the following pieces of context from PDF documents to answer the user's question.
        The context provided might be messy or contain transcription errors from the PDF. Your primary task is to synthesize this information into a single, coherent, and well-organized answer.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        INSTRUCTIONS:
        1. Provide a clear, concise, and accurate answer based ONLY on the context provided.
        2. Do NOT reproduce any garbled or duplicated text from the context. Clean it up.
        3. Use formatting such as bullet points and bold text to make the answer easy to read and professional.
        4. If the context does not contain the answer, state that you couldn't find the information in the provided documents.
        5. Do not add any information that is not present in the context.

        ANSWER:
        """
        QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=PineconeStore(
                embedding=embeddings,
                index_name=index_name,
                namespace="default"
            ).as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )

        return qa_chain, None

    except Exception as e:
        return None, f"🚨 An error occurred while building the QA system: {str(e)}"

def is_visualization_query(query: str) -> bool:
    return any(k in query.lower() for k in ["chart", "graph", "plot", "visual", "bar chart", "pie chart"])

def extract_table_from_docs(qa_chain, question: str) -> str:
    prompt = f"""
    Based on the provided documents, extract data for the following request: "{question}".
    Respond with ONLY a markdown-formatted table and nothing else. Do not include explanations, summaries, or any text outside of the table.
    The table should be suitable for direct conversion to a chart.
    """
    response = qa_chain.invoke(prompt)
    result = response.get("result", "") if isinstance(response, dict) else str(response)
    st.session_state['last_raw_table'] = result
    return result

def render_chart_from_markdown_table(md_table: str):
    try:
        table_regex = r"((?:\|.*\|(?:\r?\n|\r))+)"

        match = re.search(table_regex, md_table)
        if not match:
            st.warning("Could not find a valid markdown table in the AI's response.")
            with st.expander("Show Raw AI Output"):
                st.code(md_table)
            return

        table_str = match.group(0)
        lines = table_str.strip().splitlines()
        data_lines = [line for line in lines if "|" in line and not all(c in '-|: ' for c in line)]

        parsed_data = [[cell.strip() for cell in line.split('|') if cell.strip()] for line in data_lines]

        if len(parsed_data) < 2:
            st.warning("The extracted table has insufficient data to create a chart.")
            return

        df = pd.DataFrame(parsed_data[1:], columns=parsed_data[0])

        if len(df.columns) < 2:
            st.warning("Chart rendering failed: need at least 2 columns of data.")
            return

        x_col, y_col = df.columns[0], df.columns[1]
        df[y_col] = pd.to_numeric(df[y_col].str.replace(',', '').str.replace('[^\d.]', '', regex=True), errors='coerce')
        df.dropna(subset=[y_col], inplace=True)

        if df.empty:
            st.warning("No numeric data could be extracted for the chart's values.")
            return

        fig, ax = plt.subplots()
        df.plot(kind="bar", x=x_col, y=y_col, ax=ax, legend=False)
        plt.title(f"Chart for '{y_col}' by '{x_col}'")
        plt.ylabel(y_col)
        plt.xlabel(x_col)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to render chart: {e}")
        if 'last_raw_table' in st.session_state:
            with st.expander("Show Raw AI Output that caused the error"):
                st.code(st.session_state.get('last_raw_table'))

st.set_page_config(page_title="Document Q&A Bot", layout="wide")
st.title("📄 Document Q&A with AI + 📊 Visualization")

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if 'qa_chain' not in st.session_state: st.session_state.qa_chain = None
if 'errors' not in st.session_state: st.session_state.errors = []
if 'last_raw_table' not in st.session_state: st.session_state.last_raw_table = ""

st.header("1. Upload PDF Documents")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
process_button = st.button("Process Documents", type="primary", disabled=not uploaded_files)

if process_button:
    st.session_state.errors = []
    st.session_state.qa_chain = None
    st.cache_resource.clear()
    all_docs = read_uploaded_files(uploaded_files) if uploaded_files else []

    if all_docs:
        chain, err = create_qa_system(GOOGLE_API_KEY, PINECONE_API_KEY, all_docs, index_name="qa-pdf-index")
        if chain:
            st.session_state.qa_chain = chain
            st.success("✅ Documents processed successfully. The Q&A system is ready.")
        else:
            st.error(f"🚨 {err}")
    else:
        st.warning("⚠️ No documents were loaded. Please upload files.")

st.header("2. Ask Questions or Request Charts")
if st.session_state.qa_chain:
    user_q = st.text_input("Your Question:", placeholder="e.g., 'Summarize the financial performance' or 'Plot revenue by year'")
    if st.button("Submit", disabled=not user_q):
        with st.spinner("Thinking..."):
            try:
                if is_visualization_query(user_q):
                    st.markdown("### 📊 Extracted Table for Chart")
                    md_table = extract_table_from_docs(st.session_state.qa_chain, user_q)
                    st.markdown(md_table)
                    render_chart_from_markdown_table(md_table)
                else:
                    result = st.session_state.qa_chain.invoke(user_q)
                    answer = result.get('result', str(result))
                    st.subheader("💡 Answer")
                    st.markdown(answer)
                    if 'source_documents' in result:
                        with st.expander("📚 Sources"):
                            for src in result['source_documents']:
                                name = src.metadata.get('source', 'Unknown')
                                page = src.metadata.get('page', 'N/A')
                                st.markdown(f"**Source:** {name} (Page {page})")
                                st.caption(src.page_content[:400] + "...")
            except Exception as e:
                st.error(f"An unexpected error occurred while getting the answer: {str(e)}")
else:
    st.info("Please upload and process documents first.")
