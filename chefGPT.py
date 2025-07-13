import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import bs4
import gradio as gr

# Set USER_AGENT
os.environ["USER_AGENT"] = "cooking-assistant-chatbot"
load_dotenv()

# Groq LLM
groq_api_key = "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"
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Embeddings
os.environ['HF_TOKEN'] = "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"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load data (placeholder content; replace with cooking blogs for real data)
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    ),
)
docs = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Vectorstore
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Cooking Assistant Prompt
system_prompt = (
    "You are a helpful Cooking Assistant. "
    "Use the following pieces of retrieved context to answer "
    "the cooking-related question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise and friendly.\n\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# History-aware retriever
contextualize_q_system_prompt = (
    "Given a chat history and the latest cooking question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

from langchain.chains import create_history_aware_retriever

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Final chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Conversational chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Gradio function: display only current Q&A
def cooking_assistant(user_message, history=None):
    session_id = "gradio_cooking_session"

    if history is None:
        history = []

    # Get response from the conversational chain (history stored internally)
    response = conversational_rag_chain.invoke(
        {"input": user_message},
        config={"configurable": {"session_id": session_id}},
    )

    answer = response["answer"]

    # Only return the latest Q&A pair to UI
    gr_messages = [
        gr.ChatMessage(role="user", content=user_message),
        gr.ChatMessage(role="assistant", content=answer)
    ]

    return gr_messages

# Launch Gradio Chat Interface
gr.ChatInterface(
    fn=cooking_assistant,
    title="🍳 Cooking Assistant",
    description="Ask me cooking questions! Recipes, techniques, tips.",
    theme="soft"
).launch()
