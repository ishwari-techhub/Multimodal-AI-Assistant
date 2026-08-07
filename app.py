import streamlit as st
from modules.pdf_loader import load_pdf
from modules.txt_chunks import split_text
from modules.embedding import create_embeddings
from modules.vectorDB import create_vector_store, search_vector_store
from modules.chatbot import generate_answer , general_chat
from modules.img_chat import analyze_image
from modules.voice import record_voice,transcribe_voice,voice_chat
from modules.txt_speech import text_to_speech

st.set_page_config(
    page_title="RAG AI Chatbot",
    page_icon="🤖",
    layout="wide"
)


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

    st.title("🤖 AI Assistant")

    mode = st.radio(
        "Choose Mode",
        [
            "💬 General Chat",
            "📄 PDF Chat",
            "🖼 Image Analysis",
            "🎙️ Voice Assistant"
        ]
    )
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.session_state.general_messages = []
        st.session_state.image_messages = []
        st.rerun()

    if mode == "📄 PDF Chat":

        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"]
        )

        if st.session_state.index is not None:
            st.success("✅ PDF Loaded")
            st.write(f"Chunks : {len(st.session_state.chunks)}")
            st.write(f"Vectors : {st.session_state.index.ntotal}")

    elif mode == "💬 General Chat":
        st.success("General Chat")

    elif mode == "🖼 Image Analysis":
        st.title("🖼 Image Analysis")
        st.write("Upload an image and ask questions about it.")



    elif mode == "🔊 Text to Speech":
        st.info("Coming Soon")


if mode == "💬 General Chat":

    st.title("🤖 Multi-Modal AI Assistant")

    st.write("Ask me anything.")

    # Show previous messages
    for message in st.session_state.general_messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask anything...")

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

            with st.spinner("Thinking..."):

                answer = general_chat(question)

            st.markdown(answer)

        st.session_state.general_messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )



elif mode == "📄 PDF Chat":

    st.title("📄 PDF Chat")
    st.write("Upload a PDF and ask questions about its content.")

    if uploaded_file is not None:

        if st.session_state.index is None:

            with st.spinner("Processing PDF..."):

                text = load_pdf(uploaded_file)

                chunks = split_text(text)

                embeddings = create_embeddings(chunks)

                index = create_vector_store(embeddings)

                st.session_state.index = index
                st.session_state.chunks = chunks

            st.success("✅ PDF processed successfully!")

        # Show previous chat
        for message in st.session_state.messages:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask anything from the PDF...")

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

                with st.spinner("Searching document..."):

                    query_embedding = create_embeddings([question])

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
                with st.expander("📚 Retrieved Chunks"):

                    for idx in indices[0]:
                        st.write(st.session_state.chunks[idx])
                        st.divider()
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )
    else:

        st.info("⬅ Upload a PDF from the sidebar to start chatting.")


elif mode == "🖼 Image Analysis":

    st.title("🖼 Image Analysis")
    st.write("Upload an image and ask questions about it.")

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image:

        st.image(
            uploaded_image,
            caption="Uploaded Image",
            width="content"
        )

        # Show previous messages
        for message in st.session_state.image_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        question = st.chat_input("Ask about the image...")
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
                with st.spinner("Analyzing image..."):
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


elif mode == "🎙️ Voice Assistant":
    st.title("🎙️ Voice Assistant")
    st.write(
        "Click 🎙️, speak, and click ⏹️ when you are finished."
    )
    audio = record_voice()
    if audio:
        with st.spinner("📝 Converting speech to text..."):
            text = transcribe_voice(audio) 
        if text:
            st.subheader("📝 You said:")
            st.write(text)
            with st.spinner("🤖 Thinking..."):
                answer = voice_chat(text)
            if answer:
                st.subheader("🤖 AI")
                st.write(answer)
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
