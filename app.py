import streamlit as st

from modules.pdf_loader import load_pdf
from modules.txt_chunks import split_text
from modules.embedding import create_embeddings
from modules.vectorDB import create_vector_store, search_vector_store
from modules.chatbot import generate_answer, general_chat
from modules.img_chat import analyze_image
from modules.voice import record_voice, transcribe_voice, voice_chat
from modules.txt_speech import text_to_speech


st.set_page_config(
    page_title="NeuraAI - Multi-Modal Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 15% 15%, rgba(91,33,182,0.18), transparent 30%),
        radial-gradient(circle at 85% 20%, rgba(37,99,235,0.16), transparent 30%),
        linear-gradient(135deg, #070b17 0%, #0b1020 45%, #090d19 100%);
    color: #f8fafc;
}

.main .block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1020 0%, #0c1226 50%, #090d1b 100%);
    border-right: 1px solid rgba(139,92,246,0.20);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

.sidebar-brand {
    text-align: center;
    padding: 10px 5px 25px 5px;
}

.sidebar-logo {
    font-size: 42px;
    margin-bottom: 5px;
}

.sidebar-title {
    font-size: 25px;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sidebar-subtitle {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 4px;
}

.hero {
    padding: 25px 30px;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        rgba(124,58,237,0.18),
        rgba(37,99,235,0.12)
    );
    border: 1px solid rgba(139,92,246,0.25);
    box-shadow: 0 20px 50px rgba(0,0,0,0.30);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 38px;
    font-weight: 850;
    margin: 0;
    background: linear-gradient(90deg, #c4b5fd, #818cf8, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-text {
    color: #94a3b8;
    font-size: 15px;
    margin-top: 8px;
}

.feature-card {
    padding: 20px;
    min-height: 145px;
    border-radius: 18px;
    background: linear-gradient(
        145deg,
        rgba(30,41,59,0.65),
        rgba(15,23,42,0.65)
    );
    border: 1px solid rgba(148,163,184,0.12);
    box-shadow: 0 10px 30px rgba(0,0,0,0.20);
    transition: all 0.25s ease;
}

.feature-card:hover {
    transform: translateY(-5px);
    border-color: rgba(139,92,246,0.45);
    box-shadow: 0 15px 40px rgba(76,29,149,0.20);
}

.feature-icon {
    font-size: 30px;
}

.feature-title {
    font-size: 17px;
    font-weight: 700;
    margin-top: 8px;
}

.feature-text {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 5px;
}

[data-testid="stChatMessage"] {
    background: rgba(15,23,42,0.45);
    border: 1px solid rgba(148,163,184,0.08);
    border-radius: 18px;
    padding: 12px 16px;
    margin-bottom: 10px;
}

[data-testid="stChatInput"] {
    border-radius: 18px !important;
}

[data-testid="stChatInput"] textarea {
    background: #0f172a !important;
    color: white !important;
    border: 1px solid rgba(139,92,246,0.30) !important;
    border-radius: 16px !important;
}

[data-testid="stChatInput"] textarea:focus {
    border: 1px solid #8b5cf6 !important;
    box-shadow: 0 0 15px rgba(139,92,246,0.20) !important;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid rgba(139,92,246,0.25);
    background: linear-gradient(
        135deg,
        rgba(124,58,237,0.20),
        rgba(37,99,235,0.15)
    );
    color: #e2e8f0;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #8b5cf6;
    background: linear-gradient(
        135deg,
        rgba(124,58,237,0.40),
        rgba(37,99,235,0.30)
    );
    color: white;
    transform: translateY(-1px);
}

[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.50);
    border: 1px dashed rgba(139,92,246,0.35);
    border-radius: 16px;
    padding: 8px;
}

div[role="radiogroup"] {
    gap: 8px;
}

div[role="radiogroup"] label {
    background: rgba(15,23,42,0.60);
    border: 1px solid rgba(148,163,184,0.10);
    border-radius: 12px;
    padding: 10px 12px;
    transition: 0.2s;
}

div[role="radiogroup"] label:hover {
    border-color: rgba(139,92,246,0.45);
}

.status-card {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 15px;
    border-radius: 12px;
    background: rgba(15,23,42,0.60);
    border: 1px solid rgba(96,165,250,0.15);
    margin-top: 12px;
}

.status-dot {
    width: 9px;
    height: 9px;
    background: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 10px rgba(34,197,94,0.8);
}

.metric-card {
    padding: 16px;
    text-align: center;
    border-radius: 15px;
    background: rgba(15,23,42,0.55);
    border: 1px solid rgba(139,92,246,0.15);
}

.metric-number {
    font-size: 23px;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    color: #94a3b8;
    font-size: 12px;
}

[data-testid="stExpander"] {
    background: rgba(15,23,42,0.45);
    border: 1px solid rgba(139,92,246,0.15);
    border-radius: 14px;
}

[data-testid="stAlert"] {
    border-radius: 14px;
}

hr {
    border-color: rgba(148,163,184,0.08);
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding-top: 35px;
    padding-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "general_messages" not in st.session_state:
    st.session_state.general_messages = []

if "image_messages" not in st.session_state:
    st.session_state.image_messages = []


with st.sidebar:

    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">🤖</div>
        <div class="sidebar-title">NeuraAI</div>
        <div class="sidebar-subtitle">Multi-Modal AI Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✨ Choose Mode")

    mode = st.radio(
        "",
        [
            "💬 General Chat",
            "📄 PDF Chat",
            "🖼 Image Analysis",
            "🎙️ Voice Assistant"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    if st.button("🗑️ Clear All Chats"):
        st.session_state.messages = []
        st.session_state.general_messages = []
        st.session_state.image_messages = []
        st.rerun()

    if mode == "📄 PDF Chat":

        st.markdown("### 📚 Document")

        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            help="Upload a PDF document to chat with it."
        )

        if st.session_state.index is not None:

            st.markdown("""
            <div class="status-card">
                <div class="status-dot"></div>
                <div>
                    <b>PDF Ready</b><br>
                    <span style="color:#94a3b8;font-size:12px;">
                    Document processed successfully
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">
                            {len(st.session_state.chunks)}
                        </div>
                        <div class="metric-label">
                            Chunks
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">
                            {st.session_state.index.ntotal}
                        </div>
                        <div class="metric-label">
                            Vectors
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    elif mode == "💬 General Chat":

        st.markdown("""
        <div class="status-card">
            <div class="status-dot"></div>
            <div>
                <b>AI Online</b><br>
                <span style="color:#94a3b8;font-size:12px;">
                Ready for conversation
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif mode == "🖼 Image Analysis":

        st.markdown("""
        <div class="status-card">
            🖼️
            <div>
                <b>Vision Mode</b><br>
                <span style="color:#94a3b8;font-size:12px;">
                Image understanding enabled
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif mode == "🎙️ Voice Assistant":

        st.markdown("""
        <div class="status-card">
            🎙️
            <div>
                <b>Voice Mode</b><br>
                <span style="color:#94a3b8;font-size:12px;">
                Speech interaction enabled
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.caption("⚡ Powered by AI • RAG • Vision • Voice")


if mode == "💬 General Chat":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            🤖 Multi-Modal AI Assistant
        </div>
        <div class="hero-text">
            Ask questions, explore ideas and have an intelligent
            conversation with your AI assistant.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <div class="feature-title">Smart Conversation</div>
            <div class="feature-text">
                Ask questions and get intelligent AI-powered answers.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">AI Powered</div>
            <div class="feature-text">
                Powered by modern language models for natural responses.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Fast Responses</div>
            <div class="feature-text">
                Get answers quickly through an intuitive interface.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    for message in st.session_state.general_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("✨ Ask anything...")

    if question:

        st.session_state.general_messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("🤔 Thinking..."):
                answer = general_chat(question)

            st.markdown(answer)

        st.session_state.general_messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


elif mode == "📄 PDF Chat":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            📄 Chat With Your PDF
        </div>
        <div class="hero-text">
            Upload a document and ask questions directly from
            its content using Retrieval-Augmented Generation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is not None:

        if st.session_state.index is None:

            with st.spinner("🔄 Processing your PDF..."):

                text = load_pdf(uploaded_file)
                chunks = split_text(text)
                embeddings = create_embeddings(chunks)
                index = create_vector_store(embeddings)

                st.session_state.index = index
                st.session_state.chunks = chunks

            st.success("✅ PDF processed successfully!")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-number">
                        {len(st.session_state.chunks)}
                    </div>
                    <div class="metric-label">
                        Text Chunks
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-number">
                        {st.session_state.index.ntotal}
                    </div>
                    <div class="metric-label">
                        Embeddings
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-number">
                        RAG
                    </div>
                    <div class="metric-label">
                        Search Engine
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input(
            "🔎 Ask something about your PDF..."
        )

        if question:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):

                with st.spinner("🔍 Searching your document..."):

                    query_embedding = create_embeddings(
                        [question]
                    )

                    indices = search_vector_store(
                        st.session_state.index,
                        query_embedding,
                        k=2
                    )

                    context = ""

                    for idx in indices[0]:
                        context += (
                            st.session_state.chunks[idx]
                            + "\n\n"
                        )

                    answer = generate_answer(
                        context,
                        question
                    )

                st.markdown(answer)

                with st.expander("📚 View Retrieved Context"):

                    for idx in indices[0]:

                        st.markdown(
                            f"**Chunk {idx + 1}**"
                        )

                        st.write(
                            st.session_state.chunks[idx]
                        )

                        st.divider()

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

    else:

        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📤</div>
            <div class="feature-title">
                Upload a PDF to begin
            </div>
            <div class="feature-text">
                Select a PDF from the sidebar and start
                asking questions about your document.
            </div>
        </div>
        """, unsafe_allow_html=True)


elif mode == "🖼 Image Analysis":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            🖼️ AI Image Analysis
        </div>
        <div class="hero-text">
            Upload an image and ask questions about what
            you see. Let AI understand your visual content.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_image = st.file_uploader(
        "📤 Upload Image",
        type=["png", "jpg", "jpeg"],
        help="Upload PNG, JPG or JPEG images."
    )

    if uploaded_image:

        col1, col2 = st.columns([1, 1.2])

        with col1:

            st.image(
                uploaded_image,
                caption="Your Image",
                width="stretch"
            )

        with col2:

            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">
                    👁️
                </div>
                <div class="feature-title">
                    Vision AI Ready
                </div>
                <div class="feature-text">
                    Ask questions about objects, scenes,
                    text and other visual information.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        for message in st.session_state.image_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input(
            "🔍 Ask something about this image..."
        )

        if question:

            st.session_state.image_messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):

                with st.spinner("👁️ Analyzing image..."):

                    answer = analyze_image(
                        uploaded_image,
                        question
                    )

                st.markdown(answer)

            st.session_state.image_messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

    else:

        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">
                🖼️
            </div>
            <div class="feature-title">
                Upload an image
            </div>
            <div class="feature-text">
                Upload a JPG, JPEG or PNG image to
                start visual analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)


elif mode == "🎙️ Voice Assistant":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            🎙️ Voice AI Assistant
        </div>
        <div class="hero-text">
            Speak naturally, let AI understand your voice,
            generate an answer and convert it back to speech.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎙️</div>
            <div class="feature-title">Speak</div>
            <div class="feature-text">
                Record your voice.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Understand</div>
            <div class="feature-text">
                AI converts speech into text.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔊</div>
            <div class="feature-title">Respond</div>
            <div class="feature-text">
                AI speaks the answer back.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    audio = record_voice()

    if audio:

        with st.spinner("📝 Converting speech to text..."):
            text = transcribe_voice(audio)

        if text:

            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-title">
                        📝 You said
                    </div>
                    <div class="feature-text"
                         style="font-size:15px;color:#e2e8f0;">
                        {text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.spinner("🤖 AI is thinking..."):
                answer = voice_chat(text)

            if answer:

                st.markdown(
                    f"""
                    <div class="feature-card">
                        <div class="feature-title">
                            🤖 AI Response
                        </div>
                        <div class="feature-text"
                             style="font-size:15px;color:#e2e8f0;">
                            {answer}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                with st.spinner("🔊 Generating voice..."):
                    audio_path = text_to_speech(answer)

                if audio_path:

                    st.audio(
                        audio_path,
                        format="audio/mp3",
                        autoplay=True
                    )

            else:
                st.error("❌ Could not generate an answer.")

        else:
            st.error("❌ Could not understand the audio.")


st.markdown("""
<div class="footer">
    🤖 NeuraAI &nbsp; • &nbsp;
    RAG &nbsp; • &nbsp;
    Vision AI &nbsp; • &nbsp;
    Voice AI
</div>
""", unsafe_allow_html=True)
