import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub, HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
import requests
from langchain_community.document_loaders import WebBaseLoader
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline, BitsAndBytesConfig , CodeGenTokenizer 
from langchain.llms import HuggingFacePipeline 
from langchain import PromptTemplate, LLMChain
from transformers import AutoTokenizer , AutoModelForCausalLM
import torch 
from langchain import PromptTemplate, LLMChain

huggingface_api_key = st.secrets["HUGGINGFACEHUB_API_KEY"]

os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key


def get_vectorstore(option):
    embeddings = HuggingFaceEmbeddings(
        # model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_name="sentence-transformers/sentence-t5-xxl",
        model_kwargs={'device': 'cpu'})

    if option == "Equity, Diversion & Inclusion":
        vectorstore = FAISS.load_local("equity_diversion_inclusion_faiss", embeddings)
    elif option == "Home phone":
        vectorstore = FAISS.load_local("home_phone_faiss", embeddings)
    elif option == "Mobile phone":
        vectorstore = FAISS.load_local("mobile_phone_faiss", embeddings)
    
    return vectorstore


def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.5, "max_length":512})
    # llm = HuggingFaceHub(repo_id="microsoft/phi-2", model_kwargs={"temperature":0.5, "max_length":512})
    # llm = HuggingFaceHub(repo_id="zhengr/MixTAO-7Bx2-MoE-Instruct-v1.0", model_kwargs={"temperature":0.5, "max_length":512})
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        load_in_8bit=True,
        torch_dtype=torch.float32,
        # device_map='auto',
        quantization_config=quantization_config
    )
    pipe = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.2
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    pipe.model.config.pad_token_id = pipe.model.config.eos_token_id

    
    template = """respond to the instruction below. behave like a chatbot and respond to the user. try to be helpful.
    ### Instruction:
    {instruction}
    Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["instruction"])

    # conversation_chain = LLMChain(prompt=prompt,
    #                 llm=local_llm,
    #                 retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    #                 memory=memory
    # )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=local_llm,
        prompt=prompt,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Chat with the University",
                       page_icon=":school:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with the University :school:")
    user_question = st.text_input("Ask a question to the university  website:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Topic")
        option = st.selectbox("Please select the topic related to your query...",
                                ("Equity, Diversion & Inclusion", "Home phone", "Mobile phone"),
                                index=None,
                                placeholder="Select topic...",
        )

        st.write('You selected:', option)   

        if st.button("Process"):
            with st.spinner("Processing"):
                st.write("Processing your topic...")
                st.write("It will take a while based on the content of the topic...")
                vectorstore = get_vectorstore(option)
                print("Loaded vectorstore successfully")
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Topic processed successfully!")


if __name__ == '__main__':
    main()
