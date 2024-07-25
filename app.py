import getpass
import os
import yaml
import tempfile
import streamlit as st
from tempfile import NamedTemporaryFile
from langchain_community.vectorstores import SKLearnVectorStore
from transformers import GPT2TokenizerFast
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import ChatCohere
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from st_copy_to_clipboard import st_copy_to_clipboard


st.title("CoverMeUp")
st.markdown("**A cover letter generator based on the job description and resume using Gen AI.**")
st.divider()


uploaded_file = st.file_uploader("Upload your CV/Resume here:", type=["pdf"], key="cv-upload")
job_description = st.text_area("Job Description:", 
                               max_chars=5000, 
                               placeholder="Paste job description here", 
                               label_visibility="collapsed",
                               key="job-desc")

generator_prompt = """
Create a cover letter in the prescribed format based on the given input and context. context here refers to
the resume of the candidate while input refers to job description. 

Prescribed cover letter format:
Dear Hiring Team,
Write cover letter content that sounds professional & written by human. cover letter should have 3 - 5 paragraphs.
Sign off the cover letter as:
Best Regards,
<name in resume>

Input 1 - Resume of applying candidate: {context}
Input 2 - Job description for the job to apply: {input}
"""


@st.cache_resource
def load_embeddings():
    embeddings = HuggingFaceEmbeddings()
    return embeddings

def clear_input():
    st.session_state["job-desc"] = ""

def clear_all():
    st.session_state["cv-upload"] = ""
    st.session_state["job-desc"] = ""
    
clear_button = st.button(label="Clear Input", on_click=clear_input)
get_button = st.button("Get Cover Letter")
reload_button = st.button(label="Reload", on_click=clear_input)

if uploaded_file is not None:
    prompt_template = ChatPromptTemplate.from_template(generator_prompt)

    with st.spinner("Loading and Indexing the document"):
        
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=200, chunk_overlap=0)

        tf = NamedTemporaryFile(dir='./', suffix='.pdf', delete=False)
        tf.write(uploaded_file.getbuffer())
        loader = PDFPlumberLoader(tf.name)
        data = loader.load_and_split(text_splitter=text_splitter)
        embeddings = load_embeddings()
        persist_path = os.path.join(tempfile.gettempdir(), "union.parquet")
        vectorstore = SKLearnVectorStore.from_documents(documents=data, embedding=embeddings, persist_path=persist_path, serializer="parquet")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': len(data)})
        compressor = FlashrankRerank(top_n=1, model="ms-marco-MiniLM-L-12-v2")
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        tf.close()
        os.unlink(tf.name)
        st.info("Index created .....")
    
    with st.spinner("Initiating the LLM"):
        
        os.environ["COHERE_API_KEY"] = st.secrets["cohere-api-key"]
        cohere_llm = ChatCohere(model="command-r")  
        cohere_question_answer_chain = create_stuff_documents_chain(cohere_llm, prompt_template)
        cohere_rag_chain = create_retrieval_chain(compression_retriever, cohere_question_answer_chain)
        st.info("LLM initiated ....")


if get_button and job_description is not None:
    with st.status("Generating the cover letter"):
        input_json = {"input": job_description}
        cohere_results = cohere_rag_chain.invoke(input_json)['answer']
    st.success(cohere_results)
    st_copy_to_clipboard(cohere_results)

if reload_button:
    st.rerun()
    
