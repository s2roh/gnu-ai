# .env 파일의 API 키 정보 불러오기
import dotenv
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.document_loaders import PyMuPDFLoader

dotenv.load_dotenv()


def get_retriever(file_name, file_type):

    # 파일 로더 생성
    loader = None
    if file_type == "text/plain":
        loader = TextLoader(file_name, encoding="utf-8")
    elif file_type == "application/pdf":
        loader = PyMuPDFLoader(file_name)

    # 텍스트 파일 로딩
    if loader is not None:
        documents = loader.load()

        # splitter 생성
        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)

        # splitter로 불러온 documents를 잘라서 리스트로 만듬(자르는 크기 300, 청크마다 중복되는 내용 50)
        split_docs = text_splitter.split_documents(documents)

        # 임베딩에 사용할 임베딩 객체 생성(Openai embedding 유료)
        embedding = OpenAIEmbeddings()

        # 임베딩 객체를 이용해 문서를 임베딩한 후 벡터DB에 저장
        vectorFAISS = FAISS.from_documents(split_docs, embedding)

        # 벡터DB에서 제공하는 retriever 생성
        retrieverFAISS = vectorFAISS.as_retriever()

        return retrieverFAISS


# 화면에 사이드바 생성
file_name = ""
file_type = ""
with st.sidebar:

    # 파일 업로드 객체 생성
    uploaded_file = st.file_uploader(
        "파일을 선택하세요", type=["txt", "pdf", "xls", "xlsx"]
    )

    if uploaded_file is not None:

        # file 이름, 타입 가져오기
        file_name = "./data/" + uploaded_file.name
        file_type = uploaded_file.type

        # 파일 저장소에 파일 업로드
        with open(file_name, "wb") as f:
            f.write(uploaded_file.getbuffer())


print(f"file_name={file_name}, file_type={file_type}")

retriever = get_retriever(file_name, file_type)

query = st.chat_input("질문을 입력하세요")

if query:

    # 질의 내용을 retriever로 실행
    if retriever is not None:
        search_data = retriever.invoke(query)

        for docs in search_data:
            print("=================================")
            print(docs.page_content)
