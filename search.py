# Import necessary libraries
import streamlit as st  # For building the web app UI
from langchain_groq import ChatGroq  # LLM wrapper for Groq's models
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper  # API wrappers for Arxiv and Wikipedia
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun  # Tools to query APIs
from langchain.agents import initialize_agent, AgentType  # To initialize agents for tool-based tasks
from langchain.callbacks import StreamlitCallbackHandler  # Callback for real-time streaming output in Streamlit
import os
from dotenv import load_dotenv  # To load environment variables (if needed)

# =========================
# Initialize API Wrappers and Tools
# =========================

# Arxiv wrapper with settings: return top 1 result and max 200 characters content
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)  # Tool to run Arxiv queries

# Wikipedia wrapper with same result and character limit settings
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)  # Tool to run Wikipedia queries

# DuckDuckGo search tool
search = DuckDuckGoSearchRun(name="Search")  # Tool for web search using DuckDuckGo

# =========================
# Streamlit App Setup
# =========================

# Web page title
st.title("Search Engine with Langchain Tools & Agents")

# Sidebar input for Groq API key
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Please enter your Groq API key: ", type="password")

# =========================
# Session State Initialization for Chat History
# =========================

# If chat history is not already stored in session, initialize it
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display the current chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# =========================
# Handle User Input
# =========================

# Capture user input from the chat input box
if prompt := st.chat_input(placeholder="What is machine learning?"):
    # Save user's message in session
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)  # Display user message

    # Proceed only if API key is provided
    if api_key:
        # Initialize LLM from Groq using the provided API key
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="Llama3-8b-8192",  # Model to use
            streaming=True  # Stream output for better UX
        )

        # Combine all tools into a single list
        tools = [search, arxiv, wiki]

        # Initialize the agent to use these tools and the LLM
        search_agent = initialize_agent(
            tools,
            llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Use this agent type to dynamically decide which tool to use
            handling_parsing_error=True  # Helps prevent app from crashing on unexpected input
        )

        # =========================
        # Run Agent and Display Assistant Response
        # =========================

        # Create a new assistant message container for displaying output
        with st.chat_message("assistant"):
            # Create a callback handler to stream LLM output into Streamlit
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            # Run the agent with the full conversation history
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])

            # Append assistant response to the chat
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)  # Display the assistant's response

    else:
        # Error if no API key provided
        st.error("Please enter a valid API key.")
