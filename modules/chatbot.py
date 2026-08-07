import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
print("API KEY:", os.getenv("GROQ_API"))

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API")
)   

def generate_answer(context, question):

    prompt = f"""
    You are an AI assistant.
    Answer the question naturally using the context below.
    Do not copy the context word for word.
    Explain the answer in simple language.
    If the answer is not available, say:
    "I couldn't find the answer in the uploaded document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    response = llm.invoke(prompt)
    return response.content


def general_chat(question):
    response = llm.invoke(question)
    return response.content

