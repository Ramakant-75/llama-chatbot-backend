from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from fastapi.responses import StreamingResponse
import asyncio
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama

app = FastAPI()

# Allow frontend later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class ChatRequest(BaseModel):
    message: str

# Initialize LLM
llm = ChatOllama(model="llama3")

# Initialize memory buffer (keeps entire chat history in RAM)
memory = ConversationBufferMemory()

# Define a system prompt with a template
template = """
You are LlamaBot, a friendly, witty AI assistant developed by Ramakant Gachi. 
If anyone asks who built you, proudly say:
"I was developed by Ramakant Gachi, a passionate full-stack engineer."
You explain things clearly, use emojis occasionally, and keep responses short and helpful.
{history}
Human: {input}
AI:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template,
)

# LLM and memory setup
llm = Ollama(model="llama3", base_url="https://7afdc15e5ab0.ngrok-free.app")
memory = ConversationBufferMemory()

# ConversationChain with custom prompt
conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True,
)


@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message
    response = conversation.predict(input=user_input)
    return {"response": response}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    user_input = request.message

    async def generate():
        response = conversation.predict(input=user_input)
        for word in response.split():
            yield word + " "
            await asyncio.sleep(0.05)  # simulate natural typing

    return StreamingResponse(generate(), media_type="text/plain")