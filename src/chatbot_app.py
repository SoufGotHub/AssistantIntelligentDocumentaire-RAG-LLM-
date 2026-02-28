import streamlit as st

from chatbot_backend import ChatbotConfig, RAGChatbot


st.set_page_config(
    page_title="Assistant Documentaire RAG",
    page_icon="💬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 980px; padding-top: 1.2rem;}
    .stChatMessage {border-radius: 14px;}
    .small-muted {color: #6b7280; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=True)
def get_chatbot() -> RAGChatbot:
    cfg = ChatbotConfig()
    return RAGChatbot(cfg)


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "top_k" not in st.session_state:
        st.session_state.top_k = 4


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Paramètres")
        st.session_state.top_k = st.slider(
            "Nombre de passages (top_k)",
            min_value=1,
            max_value=8,
            value=st.session_state.top_k,
            step=1,
        )
        st.session_state.show_sources = st.toggle(
            "Afficher les sources",
            value=st.session_state.show_sources,
        )
        if st.button("Effacer la conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            sources = msg.get("sources")
            if sources and st.session_state.show_sources:
                with st.expander("Sources utilisées"):
                    for src in sources:
                        st.markdown(
                            f"- `{src['source']}` | chunk `{src['chunk_id']}` | score `{src['score']}`"
                        )
                        st.markdown(f"<div class='small-muted'>{src['preview']}</div>", unsafe_allow_html=True)


def main() -> None:
    init_state()
    render_sidebar()

    st.title("Assistant Intelligent Documentaire")
    st.caption("Posez une question sur vos documents indexés (RAG + FAISS).")

    try:
        bot = get_chatbot()
    except FileNotFoundError:
        st.error(
            "Index introuvable dans `data/vectorstore`. Lancez d'abord: "
            "`python src/build_index.py`."
        )
        return
    except Exception as exc:
        st.error(f"Erreur d'initialisation du chatbot: {exc}")
        return

    render_history()

    user_prompt = st.chat_input("Écrivez votre question...")
    if not user_prompt:
        return

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche et génération en cours..."):
            result = bot.ask(user_prompt, top_k=st.session_state.top_k)
            answer = result["answer"]
            sources = bot.format_sources(result["retrieved"])

        st.markdown(answer)
        if sources and st.session_state.show_sources:
            with st.expander("Sources utilisées"):
                for src in sources:
                    st.markdown(
                        f"- `{src['source']}` | chunk `{src['chunk_id']}` | score `{src['score']}`"
                    )
                    st.markdown(f"<div class='small-muted'>{src['preview']}</div>", unsafe_allow_html=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


if __name__ == "__main__":
    main()
