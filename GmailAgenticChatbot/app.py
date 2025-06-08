import streamlit as st
import os
import base64
from email.mime.text import MIMEText
from typing import Optional
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from langchain.tools import BaseTool

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error(" GOOGLE_API_KEY not found in .env file")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.9,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

creds = Credentials.from_authorized_user_file("token.json", ["https://www.googleapis.com/auth/gmail.send"])
gmail_service = build("gmail", "v1", credentials=creds)

class SendEmailInput(BaseModel):
    to: str = Field(..., description="Email address of the recipient")
    body: str = Field(..., description="Body of the email")

# SendEmailTool
class SendEmailTool(BaseTool):
    name: str = "send_email"
    description: str = "Send an email to someone"
    args_schema: type = SendEmailInput

    def _run(self, to: str, body: str):
        if not body.strip():
            return "❌ Error: Email body is empty."

        message = MIMEText(body)
        message['to'] = to
        message['subject'] = "Automated Email"
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        try:
            send_message = (
                gmail_service.users().messages().send(userId="me", body={"raw": raw}).execute()
            )
            return f"✅ Email sent successfully. Message ID: {send_message['id']}"
        except Exception as e:
            return f"❌ Error sending email: {e}"

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported.")

# LangChain agent setup
tool = SendEmailTool()
tools = [tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can send emails."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")  # Added this required placeholder
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Streamlit UI ---
st.set_page_config(page_title="Gmail Bot with Gemini", layout="centered")
st.title("Gmail Bot using LangChain + Gemini")

with st.form("email_form"):
    st.write("Enter the details for your email:")
    raw_input = st.text_area("Task Instruction (natural language)", value="")
    submitted = st.form_submit_button("Send Email")

if submitted:
    with st.spinner("Generating and sending email..."):
        if not raw_input.strip():
            st.error("❌ Email input is empty.")
        else:
            try:
                input_command = {"input": raw_input}
                response = agent_executor.invoke(input_command)
                st.success("✅ Done!")
                st.write("**Response:**")
                st.code(str(response), language="text")
            except Exception as e:
                st.error(f"❌ Error occurred: {str(e)}")