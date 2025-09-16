# app.py — Conversational RAG (Text + Mic + WAV) with chat memory

import os
import tempfile
from typing import List

import streamlit as st
from dotenv import load_dotenv

# PDF reader (pypdf)
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# LangChain / Vector store / prompts / chains / messages
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

# Voice I/O
import speech_recognition as sr
import pyttsx3
import uuid


# =========================
# Config & page
# =========================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY in your environment.")

# You can change models here if desired
LLM_MODEL = "gemma2-9b-it"  # e.g., "mixtral-8x7b-instruct", "llama-3.1-70b-versatile"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(
    page_title="AI Automobile Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚗 AI-Powered Automobile Assistant (RAG, Conversational)")
st.caption("Upload a manual (PDF) → Process → Ask questions via text or voice, grounded in your document.")


# =========================
# Session State (init)
# =========================
defaults = {
    # Index
    "vectordb": None,
    "processed_filename": None,
    "ready": False,

    # Text-mode input
    "query_text": "",

    # Voice/WAV input
    "recognized_text": "",
    "pending_recognized_text": None,
    "prev_mode": "Text",
    "wav_version": 0,  # versioned uploader key to avoid rerun loops

    # Chat memory
    "messages": [],  # list[HumanMessage | AIMessage]

    # Output & TTS
    "last_answer": "",
    "tts_path": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================
# Utilities
# =========================
def read_pdf_text(path: str, password: str | None = None) -> str:
    """Read all pages from a (possibly encrypted) PDF and return concatenated text."""
    try:
        reader = PdfReader(path)
    except PdfReadError as e:
        raise RuntimeError(f"PDF read error: {e}")

    if getattr(reader, "is_encrypted", False):
        tried = []
        for pwd in [password, ""]:
            if pwd is None:
                continue
            tried.append("<provided>" if pwd else "<empty>")
            try:
                res = reader.decrypt(pwd)  # 0=failure, 1=user pw, 2=owner pw
            except Exception:
                res = 0
            if res:
                break
        else:
            raise RuntimeError(
                f"Encrypted PDF. Could not decrypt with {', '.join(tried)}. "
                "Provide the correct password."
            )

    parts: List[str] = []
    for p in reader.pages:
        t = p.extract_text() or ""
        if t:
            parts.append(t)
    return "\n".join(parts)


def split_text(text: str, chunk_size=1200, chunk_overlap=150) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def build_vectordb(chunks: List[str]):
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    # session-only Chroma (no persist) for a clean run
    return Chroma.from_texts(chunks, embedding=embeddings)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(groq_api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=0.2)


def synthesize_tts(text: str) -> str:
    """Save TTS to a temp WAV and return the path."""
    engine = pyttsx3.init()
    out_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.wav")
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    return out_path


def reset_app():
    st.session_state.vectordb = None
    st.session_state.processed_filename = None
    st.session_state.ready = False

    st.session_state.query_text = ""
    st.session_state.recognized_text = ""
    st.session_state.pending_recognized_text = None
    st.session_state.prev_mode = "Text"
    st.session_state.wav_version = 0

    st.session_state.messages = []
    st.session_state.last_answer = ""
    st.session_state.tts_path = ""


# =========================
# Conversational RAG chain builders
# =========================
# 1) Contextualize follow-ups into standalone queries (history-aware retriever)
CONTEXTUALIZE_Q_SYSTEM = (
    "Given the chat history and the latest user question, rewrite the question so it is "
    "fully self-contained and unambiguous without needing the prior messages. "
    "Do NOT answer the question; only rewrite it if necessary, otherwise return it as is."
)

# 2) Answer using retrieved context + chat history (no hallucination)
QA_SYSTEM = (
    "You are an expert assistant for automobile manuals. "
    "Use ONLY the retrieved context to answer the user's question. "
    "Be concise and factual. If the answer is not present in the provided context, say "
    "\"I don't know based on the manual provided.\" Do not speculate.\n\n"
    "Context:\n{context}"
)

def build_history_aware_chain(vectordb, k: int):
    """
    Returns a chain that:
      - uses an LLM to rewrite follow-up questions via chat history,
      - retrieves top-k chunks,
      - answers grounded only in the retrieved context + history.
    """
    llm = get_llm()

    # fresh retriever with current top-k
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_Q_SYSTEM),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain


# =========================
# SIDEBAR — Step 1 & 2 (Upload + Process)
# =========================
with st.sidebar:
    st.header("Step 1 — Upload your manual (PDF)")
    uploaded_pdf = st.file_uploader("Drop a PDF here", type=["pdf"], key="pdf_upload")
    pdf_password = st.text_input("PDF password (optional)", type="password", key="pdf_pw")

    st.header("Step 2 — Process")
    process_clicked = st.button(
        "Process PDF",
        type="primary",
        use_container_width=True,
        disabled=uploaded_pdf is None,
        key="process_btn",
    )

if process_clicked and uploaded_pdf is not None:
    tmp_dir = tempfile.gettempdir()
    save_path = os.path.join(tmp_dir, uploaded_pdf.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    with st.spinner("Extracting text from PDF..."):
        try:
            text = read_pdf_text(save_path, password=st.session_state.pdf_pw)
        except Exception as e:
            st.sidebar.error(str(e))
            st.stop()

        if not text.strip():
            st.sidebar.error("No selectable text found (scanned PDF?). Try an OCR'ed PDF.")
            st.stop()

    with st.spinner("Splitting into chunks..."):
        chunks = split_text(text, chunk_size=800, chunk_overlap=150)
        if not chunks:
            st.sidebar.error("Failed to create text chunks.")
            st.stop()

    with st.spinner("Embedding and indexing (Chroma)..."):
        try:
            vectordb = build_vectordb(chunks)
        except Exception as e:
            st.sidebar.error(f"Vector DB build failed: {e}")
            st.stop()

    st.session_state.vectordb = vectordb
    st.session_state.processed_filename = uploaded_pdf.name
    st.session_state.ready = True
    st.session_state.messages = []  # fresh chat for new manual
    st.sidebar.success("PDF processed successfully ✅")


# =========================
# MAIN — Step 3 (Conversational Q&A)
# =========================
st.subheader("Step 3 — Ask questions (conversational)")

if not st.session_state.ready:
    st.info("⬅️ Upload and process a PDF from the sidebar to start.")
else:
    # Show chat transcript
    if st.session_state.messages:
        for m in st.session_state.messages:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(m.content)

    # Input method choice
    mode = st.radio(
        "Choose input method",
        ["Text", "Voice (microphone)", "Upload WAV file"],
        horizontal=True,
        key="input_mode",
    )

    # Clear cross-mode artifacts on switch (audio & stale inputs)
    if mode != st.session_state.prev_mode:
        if mode == "Text":
            st.session_state.recognized_text = ""
        else:
            st.session_state.query_text = ""
        st.session_state.tts_path = ""
        st.session_state.last_answer = ""
        st.session_state.prev_mode = mode

    # Apply any pending recognized text before rendering widgets
    if st.session_state.pending_recognized_text is not None:
        st.session_state.recognized_text = st.session_state.pending_recognized_text
        st.session_state.pending_recognized_text = None

    # ---- Render inputs per mode ----
    user_query = ""
    if mode == "Text":
        # Use chat-style input for text mode
        user_query = st.chat_input("Your question:")
        if user_query is None:
            user_query = ""  # no submit yet

    elif mode == "Voice (microphone)":
        if st.button("🎤 Record from mic (5 sec)", key="mic_record"):
            recognizer = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    st.info("Listening for 5 seconds... please speak clearly")
                    audio = recognizer.record(source, duration=5)
                try:
                    rec = recognizer.recognize_google(audio)
                    st.session_state.pending_recognized_text = rec
                    st.success(f"Recognized: {rec}")
                    st.rerun()
                except sr.UnknownValueError:
                    st.error("Could not understand audio.")
                except sr.RequestError as e:
                    st.error(f"Speech-to-text service failed: {e}")
            except Exception as e:
                st.error(f"Microphone error: {e}\nIf microphone is unavailable, try 'Upload WAV file'.")
        # show recognized
        if st.session_state.recognized_text:
            st.caption(f"Recognized: “{st.session_state.recognized_text}”")
        else:
            st.caption("Recognized: —")
        # Send button for voice
        send_voice = st.button("Send", key="send_voice_btn", disabled=not st.session_state.recognized_text.strip())
        if send_voice:
            user_query = st.session_state.recognized_text.strip()

    else:  # Upload WAV file
        # Versioned key prevents infinite rerun by clearing the file selection after we process it
        wav_key = f"wav_upload_{st.session_state.wav_version}"
        wav_file = st.file_uploader("Upload WAV (16kHz mono preferred)", type=["wav"], key=wav_key)
        if wav_file is not None:
            tmp_wav = os.path.join(tempfile.gettempdir(), wav_file.name)
            with open(tmp_wav, "wb") as f:
                f.write(wav_file.getbuffer())
            recognizer = sr.Recognizer()
            try:
                with sr.AudioFile(tmp_wav) as source:
                    audio = recognizer.record(source)
                rec = recognizer.recognize_google(audio)
                st.session_state.pending_recognized_text = rec
                st.success(f"Recognized from WAV: {rec}")
                # Bump version so the uploader key changes → clears uploaded file → breaks rerun loop
                st.session_state.wav_version += 1
                st.rerun()
            except sr.UnknownValueError:
                st.error("Could not understand uploaded audio.")
            except sr.RequestError as e:
                st.error(f"Speech recognition failed: {e}")
        if st.session_state.recognized_text:
            st.caption(f"Recognized: “{st.session_state.recognized_text}”")
        else:
            st.caption("Recognized: —")
        send_wav = st.button("Send", key="send_wav_btn", disabled=not st.session_state.recognized_text.strip())
        if send_wav:
            user_query = st.session_state.recognized_text.strip()

    # Retrieval control
    k = st.number_input("Top-K context chunks", min_value=1, max_value=10, value=3, step=1, key="topk")

    # Handle a new user query (from any mode)
    if user_query:
        # Append user message
        st.session_state.messages.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        # Build chain with current k and run with history
        rag_chain = build_history_aware_chain(st.session_state.vectordb, k)
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.messages[:-1]  # history excludes current user turn
            })

        answer = response.get("answer") if isinstance(response, dict) else str(response)

        # Append assistant message
        st.session_state.messages.append(AIMessage(content=answer))
        with st.chat_message("assistant"):
            st.markdown(answer)
            # Optional: show retrieved context
            ctx_docs = response.get("context", []) if isinstance(response, dict) else []
            if ctx_docs:
                with st.expander("Show retrieved context (top matches)"):
                    for i, d in enumerate(ctx_docs, start=1):
                        st.markdown(f"**Match {i}**")
                        st.write(d.page_content[:2000])

        # Persist last answer (for TTS)
        st.session_state.last_answer = answer

        # Clear recognized text after sending (voice/wav)
        if mode != "Text":
            st.session_state.recognized_text = ""

    # Speak / Reset row
    cols = st.columns([1, 1])
    with cols[0]:
        speak = st.button("🔊 Speak Answer", key="speak_btn")
    with cols[1]:
        reset = st.button("Reset", on_click=reset_app, key="reset_btn")

    if speak:
        if not st.session_state.get("last_answer"):
            st.warning("No answer to speak yet. Ask something first.")
        else:
            try:
                st.session_state.tts_path = synthesize_tts(st.session_state.last_answer)
                st.success("Audio ready below. Press play ▶️")
            except Exception as e:
                st.error(f"TTS Error: {e}")

    if st.session_state.get("tts_path"):
        st.audio(st.session_state.tts_path)

    # Footer
    st.caption(f"Indexed file: **{st.session_state.processed_filename}**")
