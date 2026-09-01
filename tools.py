import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag import retriever, chain as rag_chain

load_dotenv()

general_model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")
general_prompt = ChatPromptTemplate.from_messages(
    [
    (
    "system",
    """Sen genel soruları cevaplayan bir yapay zeka asistanısın"""
    ),
    (
    "human", 
    "{question}"
    )
    ]
)

general_chain = general_prompt | general_model | StrOutputParser()

@tool
def general_question_tool(question: str) -> str:
    """
    Genel bilgi, günlük yaşam, yazılım, yemek, teknoloji, eğtiim ve MIL-STD teknik dökümanıyla ilgisi olmayan diğer kullanıcı sorularını cevaplamak için kullanılır.
    """

    response = general_chain.invoke({"question": question})

    return response

@tool
def mil_std_rag_tool(question: str) -> str:
    """
    MIL-STD teknik dökümanıyla ilgili kullanıcı sorularını cevaplamak için kullanılır.
    Çalışma sıcaklığı, depolama sıcaklığı, nem, titreşim, mekanik şok, rakım ve güç gereksinimleri ve test kriterleri gibi teknik döküman sorularında bu tool kullanılır.
    """

    documents = retriever.invoke(question)
    context = "\n\n".join(document.page_content for document in documents)
    response = rag_chain.invoke({"context": context, "question": question})

    return response