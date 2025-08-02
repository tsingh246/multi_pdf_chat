import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models  import ChatOpenAI
from htmlTemplates import bot_template, user_template, css




# from langchain.text_splitter import CharacterTextSplitter
def get_pdf_text(pdf_files):
    text= ""
    for pdf_file in pdf_files:
        # Read the PDF file
        pdf_reader = PdfReader(pdf_file)
        # Extract text from each page
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def get_text_chunks(text):
    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    # Split the text into chunks
    text_chunks = text_splitter.split_text(text)
    return text_chunks    

def create_vector_store(text_chunks):
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    # Create a vector store using FAISS
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    # Initialize conversation memory
    # This memory will store the chat history
    conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
        )
    # Create a conversational retrieval chain
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=conversation_memory
    )
    return conversation

def handle_user_question(user_question):
    

    if st.session_state.conversation is None:
        st.error("Please upload and process PDFs first.")
        return

    # Get the conversation chain
    conversation = st.session_state.conversation

    # Get the response from the conversation chain
    response = conversation({"question": user_question})

    # Display the response
    #st.write("Response:", response['answer'])
    
    st.session_state.chat_history = response['chat_history']
  
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
    



def main():
    load_dotenv()
    st.set_page_config(page_title="Multi PDF Chat", page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)
   
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Multi PDF Chat Application")

    st.write("This is a simple application to chat with multiple PDFs.")
    user_question = st.text_input("Ask your question about your documents here:")
   
    if user_question:
        handle_user_question(user_question)
   
    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs=st.file_uploader("Choose PDF files", accept_multiple_files=True, type=["pdf"])
        print("get pdf fpcs **********",pdf_docs)
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                if pdf_docs:
                    for pdf in pdf_docs:
                        st.write(f"Uploading {pdf.name}...")

                        st.success(f"{pdf.name} uploaded successfully!")
                else:
                    st.error("Please upload at least one PDF file.")
                
                #Get PDF text content
                raw_text = get_pdf_text(pdf_docs)
                #st.write("Extracted Text:",raw_text)
            
                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write("Text Chunks:", text_chunks)
                 # Create vector store
                vector_store = create_vector_store(text_chunks)   
                st.write("Vector Store Created Successfully!")
                st.write("Vector Store:", vector_store)
                #st.write(FAISS.get_by_ids(vector_store, [0, 1, 2]))  # Example to get first three vectors
                # Create conversation with the vector store

                st.session_state.conversation = get_conversation_chain(vector_store)

        
    
 
if __name__ == "__main__":
    main()
