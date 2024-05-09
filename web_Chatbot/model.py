from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import getpass
import os
from langchain_openai import ChatOpenAI,OpenAI
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate


def ChatTemplate():
  """Instructs the model to answer a question in a detailed way, citing the source document only at the end.

  This function creates a prompt for GPT instruct models that includes:
      - Instruction to answer the question using the provided document ONLY.
      - Requirement for detailed answers that go beyond simple yes/no responses.
      - Instruction to **ONLY cite the source document in square brackets as '[source]' at the very end of the answer**, if the answer is found within the context.
      - Handling of cases where the answer cannot be found in the document, with a specific response to indicate this.

  Args:
      None

  Returns:
      A ChatPromptTemplate object containing the formatted prompt.
  """

  return ChatPromptTemplate.from_messages(
      [
          ("system", "I will provide you with a context passage enclosed in triple quotes and a question. Your task is to answer the question in a comprehensive and informative way, using the provided context as your ONLY reference. If the answer can be found from the information in the passage, insert the text at the very end of your answer. Otherwise, respond with 'Insufficient information.'.Use the following format to insert source text: [\"Source\": …]."),
          ("human", "\"\"\"{context} \"\"\"\n\nQuestion: {question}")
      ]
  )





llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=2,max_tokens=1024)



class RAG:
    def __init__(self,url):
        self.url=url
        self.loader=WebBaseLoader(url)
        self.data=self.loader.load()
        
    def get_vectors(self):
        data=self.data
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(data)
        self.vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        return self.vectorstore
    def retrieval(self):
        vectorstore = self.get_vectors()
        retriever = vectorstore.as_retriever()
        return retriever
    def predict(self,question):
        chat_template=ChatTemplate()
        url=self.url
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        retriever = self.retrieval()
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | chat_template
            | llm
            | StrOutputParser()
        )
           
        answer =  rag_chain.invoke(question)
        self.vectorstore.delete_collection()
        return answer
  
        
        

    
# data=RAG("https://en.wikipedia.org/wiki/Generative_artificial_intelligence")

# print(data.predict('What is football?'))



# print(answer)
# # vectorstore=data.get_vectors()
# # # retriever = vectorstore.as_retriever()
# prompt = hub.pull("aman8t/gg")
# print(prompt)


# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)



# )
# rag_chain_with_source = RunnableParallel(
#     {"context": retriever, "question": RunnablePassthrough()}
# ).assign(answer=rag_chain_from_docs)

# print(rag_chain.invoke("What is What are the concerns around Generative AI"))
# vectorstore.delete_collection()

