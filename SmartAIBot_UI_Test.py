import torch
import streamlit as st
from smartai import load_model
from langchain.vectorstores import Chroma
from const import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME
from langchain.chains import RetrievalQA
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import utils  # ✅ ensure consistency with smartai.py


def model_memory():
    # Improved, stricter grounding prompt
    template = """Use the following pieces of context to answer the question at the end.
If the answer cannot be found in the context, say "I cannot find this information in the provided documents."
Do not use any prior knowledge outside the context.

{context}

{history}
Question: {question}
Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return prompt, memory


# Sidebar contents
with st.sidebar:
    st.title("🤗💬 Converse with your Data")
    st.markdown(
        """
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [SmartAI_Bot](https://github.com/Ganavi13/SmartAI_Bot)
    """
    )
    add_vertical_space(5)
    st.write("Made by Ganavi C S")


# Detect device
if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"


# --- Initialize Session State ---
if "EMBEDDINGS" not in st.session_state:
    # ✅ use utils.get_embeddings() for consistency with your smartai.py setup
    EMBEDDINGS = utils.get_embeddings(DEVICE_TYPE)
    st.session_state.EMBEDDINGS = EMBEDDINGS

if "DB" not in st.session_state:
    DB = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=st.session_state.EMBEDDINGS,
        client_settings=CHROMA_SETTINGS,
    )
    st.session_state.DB = DB

if "RETRIEVER" not in st.session_state:
    # ✅ increase k to improve retrieval relevance
    RETRIEVER = st.session_state.DB.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    st.session_state.RETRIEVER = RETRIEVER

if "LLM" not in st.session_state:
    LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
    st.session_state.LLM = LLM

if "QA" not in st.session_state:
    prompt, memory = model_memory()
    QA = RetrievalQA.from_chain_type(
        llm=st.session_state.LLM,
        chain_type="stuff",
        retriever=st.session_state.RETRIEVER,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    st.session_state.QA = QA


# --- UI Section ---
st.title("🤖 Smart AI Bot")

temp_query_input = st.text_input("Input your prompt here", key="query_input", value="")

if st.button("Submit Query"):
    if temp_query_input:
        response = st.session_state.QA(temp_query_input)
        answer, docs = response["result"], response["source_documents"]

        st.write(answer)

        with st.expander("Document Similarity Search"):
            search = st.session_state.DB.similarity_search_with_score(temp_query_input)
            for i, doc in enumerate(search[:3]):  # ✅ show top 3 docs only
                st.write(f"Source Document #{i+1}: {doc[0].metadata['source'].split('/')[-1]}")
                st.write(doc[0].page_content)
                st.write("--------------------------------")
