from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API"))

models = client.models.list()

print("Available models:\n")

for model in models.data:
    print(model.id)