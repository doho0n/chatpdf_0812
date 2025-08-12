import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader

# 1) PDF → 텍스트 변환
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# 2) FAISS 벡터스토어 생성
def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v1")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# 3) LLM 초기화
@st.cache_resource
def load_openai_llm():
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY를 환경변수나 Streamlit secrets에 설정하세요.")
        st.stop()
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=api_key)

# 4) Streamlit UI
def main():
    st.set_page_config(page_title="📄 ChatPDF", page_icon="📄")
    st.title("📄 ChatPDF (RAG)")

    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])
    if uploaded_file:
        with st.spinner("PDF에서 텍스트 추출 중..."):
            text = extract_text_from_pdf(uploaded_file)
            vectorstore = build_vectorstore(text)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        llm = load_openai_llm()

        # 채팅 세션 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 이전 대화 표시
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 사용자 입력
        user_input = st.chat_input("PDF 내용에 대해 질문해보세요.")
        if user_input:
            st.chat_message("user").markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            docs = retriever.invoke(user_input)
            if not docs:
                answer = "문서에서 관련 내용을 찾지 못했습니다."
            else:
                context = "\n\n".join([d.page_content for d in docs])
                prompt = f"다음 PDF 내용만 바탕으로 질문에 답하세요.\n\n{context}\n\n질문: {user_input}\n답변:"
                answer = llm.invoke(prompt).content.strip()

            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
