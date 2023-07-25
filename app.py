import streamlit as st
import langchain
langchain.verbose = False

from dotenv import load_dotenv
from PyPDF2 import PdfReader
# from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from htmlTemplates import css, bot_template, user_template

# https://www.youtube.com/watch?v=dXxQ0LR-3Hg
def main():
  load_dotenv()
  createFront()
  

def createFront(): 
  # Config streamlit (front)
  st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:", layout="wide")
  st.write(css, unsafe_allow_html=True)
  
  # Create a session variable. Initialize it to None if not exists.
  if "conversation" not in st.session_state:
    st.session_state.conversation = None
  if "chatHistory" not in st.session_state:
    st.session_state.chatHistory = None
    
  ## Main Section 
  st.header("Chat with multiple PDFs :books:")
  userInput= st.text_input("Ask a question about your documents:", key="name")
  if userInput: handleUserInput(userInput)
  
  ## Sidebar Section
  with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process"):
      with st.spinner("Processing..."):
        processPDFs(pdf_docs) 
        

## Region Handle User Input
def handleUserInput(userInput):
  response = st.session_state.conversation({'question': userInput})
  
  st.session_state.chatHistory = response['chat_history']
  
  for i, message in enumerate(st.session_state.chatHistory):
    if i % 2 == 0:
      st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
      # message starts with HumanMessage
      st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

## Region Process PDFs, Get Chunks, Create Embeddings, Create Conversation Chain
def processPDFs(pdf_docs):
  #1 get PDF text
  raw_text = getPdfText(pdf_docs)
  #2 get text chunks
  text_chunk = getTextChunks(raw_text)
  #3 create Embeddings and vector store
  vectorStore = getVectorStore(text_chunk)
  
  #4 Create conversation chain
  st.session_state.conversation = getConversationChain(vectorStore)       
  
def getPdfText(pdf_docs):
  text = ""
  for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      text += page.extract_text()
  return text

def getTextChunks(text):
  splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
  text_chunks = splitter.split_text(text)
  return text_chunks

def getVectorStore(text_chunks):
  embedding = OpenAIEmbeddings()
  # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
  
  vector_store = FAISS.from_texts(texts=text_chunks, embedding=embedding)
  return vector_store

def getConversationChain(vector_store):
  # llm = OpenAI() #davinci
  llm = ChatOpenAI()
  
  #HuggingFaceHub. huggingface.com/models  
  llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
  # Memory
  memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, vector_store=vector_store, max_memory_size=1000) 
  conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
  
  return conversation_chain 
## End Region Process PDFs, Get Chunks, Create Embeddings, Create Conversation Chain

if __name__ == '__main__':
    main()