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


st.title("Cover Me")
st.markdown("**Do you want to stand out while applying for jobs? We got you covered! Cover Me generates contextual cover letter based on your resume & the job description in seconds. You can now ditch your generic messages & send custom cover letters to boost your chances of converting.**")
st.markdown("Note: We're in the early stages of development. Please bear with us and help us improve with your feedback. Send your suggestions and feature requests to roshan2971@gmail.com.")
st.divider()


uploaded_file = st.file_uploader("Upload your CV/Resume here:", type=["pdf"], key="cv-upload")
job_description = st.text_area("Job Description:", 
                               max_chars=5000, 
                               placeholder="Paste job description here", 
                               label_visibility="collapsed",
                               key="job-desc")

generator_prompt = """
Create a cover letter in the Format structure given below. Use the given 2 inputs below as variable to add context to the format.
Format structure of cover letter:
Note - Different section of the letter is given in </> tags for your internal understanding. Follow the instruction within the tags for each sections. Don’t generate the tags in the output. Line after # is a strict command to be followed. 
<Salutation>
Dear Hiring Team,
</Salutation>
<Body>
Write cover letter body that sounds professional & written by human. cover letter should strictly have 4 paragraphs (Around 250 words max). The body should not sound blatantly related job description. Make it subtle by following the below steps
Step 1: Extract important key words from input 2 given below
Step 2: Extract details about the company from input 2 given below + internet (vision, key products etc.)
Step 3: Use extraction from step 1 & step 2 to combine with input 1 given below as context to write cover letter
Don’t use terms like “as mentioned in the job description”
</Body>
<Signature>
# Sign off the cover letter with 2 lines given below:
<Line 1>Best Regards,</Line 1>  
<Line 2> Name in resume (input 2)</Line 2>
</Signature>
Input 1 - Resume of applying candidate: {context}
Input 2 - Job description for the job to apply: {input}
Note: The output should only contain the cover letter as given in the format structure
"""


@st.cache_resource
def load_embeddings():
    embeddings = HuggingFaceEmbeddings()
    return embeddings

def clear_input():
    st.session_state["job-desc"] = ""

def clear_all():
    uploaded_file = None
    st.session_state["job-desc"] = ""
    
clear_button = st.button(label="Clear Input", on_click=clear_input)
get_button = st.button("Get Cover Letter")
reload_button = st.button(label="Reload", on_click=clear_all)

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
    
