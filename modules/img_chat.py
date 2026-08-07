import os
import base64
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    api_key=os.getenv("GROQ_API")
)

def encode_image(uploaded_file):
    image_bytes = uploaded_file.read()
    return base64.b64encode(image_bytes).decode("utf-8")

def analyze_image(uploaded_file, question):
    image = encode_image(uploaded_file)
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": question
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }
            }
        ]
    )

    response = llm.invoke([message])
    return response.content