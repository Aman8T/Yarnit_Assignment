from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import getpass
import os
from langchain_openai import ChatOpenAI,OpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from model import get_examples,ContentAi
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
print()
class Response(BaseModel):
    result: str | None

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#url: str,question: str
llm = OpenAI(model="gpt-3.5-turbo",temperature=0.1,max_tokens=1024)

@app.post("/predict", response_model=Response)
def predict(payload: dict) -> Any:
    try:
        format = payload.get("format", "")
        topic = payload.get("topic", "")
        if not format or not topic:
            raise HTTPException(status_code=422, detail="Missing 'format' or 'topic'")
        
        
        content=ContentAi(format,topic)
        answer= content.generate()
        
        
        return {"result": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
