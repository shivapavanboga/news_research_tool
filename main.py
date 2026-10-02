import os
import math
import re
import streamlit as st
import pickle
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google.api_core.exceptions import ResourceExhausted
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = "gemini-3.5-flash"
# The free tier allows only ~20 requests per day *per model*, so fall back through
# these in order when one is exhausted. Each has its own independent allowance.
FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview"]
EMBEDDING_MODEL = "models/gemini-embedding-001"


class PicklableGeminiEmbeddings(GoogleGenerativeAIEmbeddings):
    """Drops the gRPC channel when pickling and rebuilds it on load.

    The FAISS index is saved to disk, and FAISS stores the embedding function
    alongside it. The default client holds a live gRPC channel, which cannot be
    pickled, so saving the index would fail with
    "no default __reduce__ due to non-trivial __cinit__".
    """

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("client", None)
        return state

    def __setstate__(self, state):
        object.__setattr__(self, "__dict__", state)
        object.__setattr__(self, "__fields_set__", set(self.__fields__.keys()))
        self.client = self.validate_environment(dict(self.__dict__))["client"]


def ask_with_fallback(question, retriever):
    """Try each model in turn, moving on if one is rate limited.

    Raises the last ResourceExhausted only when every model is exhausted.
    """
    last_error = None
    for model_name in [LLM_MODEL] + FALLBACK_MODELS:
        try:
            # Low temperature: this is factual extraction from the supplied articles,
            # and the course's original 0.9 setting made models answer "I don't know".
            llm = ChatGoogleGenerativeAI(
                model=model_name, temperature=0.0, max_output_tokens=500
            )
            chain = RetrievalQAWithSourcesChain.from_llm(llm, retriever=retriever)
            return chain.invoke({"question": question}, return_only_outputs=True)
        except ResourceExhausted as e:
            last_error = e
            continue
    raise last_error


st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_gemini.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=[u for u in urls if u.strip()])
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    if not data:
        st.error("Could not load any article. Check the URLs are reachable and try again.")
        st.stop()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    if not docs:
        st.error("Articles loaded but produced no text to index.")
        st.stop()
    # create embeddings and save it to FAISS index
    embeddings = PicklableGeminiEmbeddings(model=EMBEDDING_MODEL)
    vectorstore_gemini = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_gemini, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            try:
                result = ask_with_fallback(query, vectorstore.as_retriever())
            except ResourceExhausted as e:
                # Every model is rate limited. The message carries the server's own
                # retry hint, so surface that instead of a raw traceback.
                match = re.search(r"retry in ([\d.]+)s", str(e))
                wait = f"about {math.ceil(float(match.group(1)))} seconds" if match else "a minute"
                st.warning(
                    "The Gemini free-tier quota is exhausted for every model this app "
                    f"tries. Please retry in {wait}."
                )
                st.stop()
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = [s for s in sources.split("\n") if s.strip()]
                for source in sources_list:
                    st.write(source)




