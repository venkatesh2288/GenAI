import streamlit as st
import pandas as pd
import os
import io
import sys
import tempfile
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType

# Streamlit App Title
st.title("📊 CSV Chatbot using LangChain + Gemini")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "csv_agent" not in st.session_state:
    st.session_state.csv_agent = None

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

# Process uploaded file
if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            st.session_state.tmp_file_path = tmp_file_path

        # Initialize Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key="AIzaSyDGOdBnd43a8YKza0V3jzKBnCdedoUjVH0",  # Replace with your actual API key
            temperature=0
        )

        # Create CSV agent
        st.session_state.csv_agent = create_csv_agent(
            llm,
            tmp_file_path,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True
        )

        st.success(f"CSV file '{uploaded_file.name}' uploaded successfully!")

        # Display basic info
        df = pd.read_csv(uploaded_file)
        st.write(f"**File Info:** {len(df)} rows × {len(df.columns)} columns")

    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")

# Chat interface
if st.session_state.csv_agent is not None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask a question about your CSV data"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Capture print() output using StringIO
                    old_stdout = sys.stdout
                    redirected_output = sys.stdout = io.StringIO()

                    response = st.session_state.csv_agent.run(prompt)

                    sys.stdout = old_stdout
                    printed_output = redirected_output.getvalue().strip()

                if printed_output:
                    st.write("```\n" + printed_output + "\n```")

                st.write(response)

            full_response = printed_output + "\n" + response if printed_output else response
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            error_message = f"Error: {str(e)}"
            with st.chat_message("assistant"):
                st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    st.info("Please upload a CSV file to start chatting about your data.")

# Cleanup on rerun
if hasattr(st.session_state, 'tmp_file_path'):
    try:
        os.unlink(st.session_state.tmp_file_path)
    except:
        pass
