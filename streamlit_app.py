from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from app.graph.workflow import app_graph


AGENT_LABELS = {
    "coding": "Coding agent",
    "general": "General agent",
    "planning": "Planning agent",
    "research": "Research agent",
}


def create_conversation() -> str:
    conversation_id = str(uuid4())
    st.session_state.conversations[conversation_id] = {
        "title": "New conversation",
        "messages": [],
        "history": [],
        "error": None,
    }
    st.session_state.current_conversation_id = conversation_id
    return conversation_id


def submit_message(query: str, conversation: dict) -> None:
    history = conversation["history"]

    try:
        result = app_graph.invoke(
            {
                "query": query,
                "original_query": query,
                "intent": "",
                "draft": "",
                "review": "",
                "answer": "",
                "retry_count": 0,
                "history": history.copy(),
            },
            config={
                "configurable": {
                    "thread_id": st.session_state.current_conversation_id,
                }
            },
        )
        answer = result.get("answer", "").strip()

        if not answer:
            raise ValueError("The agent returned an empty response.")

        conversation["error"] = None
        intent = result.get("intent", "")
        metrics = []
        latency = result.get("ollama_latency_seconds")
        tokens = result.get("ollama_total_tokens")
        if latency is not None:
            metrics.append(f"{latency}s")
        if tokens is not None:
            metrics.append(f"{tokens} tokens")

        conversation["messages"].extend([
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": answer,
                "agent": AGENT_LABELS.get(intent, "Assistant"),
                "metrics": " · ".join(metrics),
            },
        ])
        history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ])
        conversation["history"] = history[-10:]

        if conversation["title"] == "New conversation":
            conversation["title"] = query[:42] + ("…" if len(query) > 42 else "")
    except Exception as exc:
        conversation["error"] = str(exc)


def main() -> None:
    st.set_page_config(
        page_title="Agentic Chat",
        page_icon="✦",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(ellipse at 50% -18%, #24334e 0, transparent 42%),
                #0b0e14;
        }
        [data-testid="stSidebar"] {
            background: #10141d;
            border-right: 1px solid #252b38;
        }
        .hero {
            padding: 2.25rem 0 1.25rem;
            text-align: center;
        }
        .hero-mark {
            color: #a78bfa;
            font-size: 2rem;
        }
        .hero h1 {
            margin: .3rem 0;
            font-size: 2.35rem;
            letter-spacing: -.055em;
        }
        .hero p {
            color: #9ba4b5;
            font-size: 1rem;
        }
        [data-testid="stChatMessage"] {
            border: 1px solid #252b38;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: rgba(20, 25, 35, .72);
        }
        [data-testid="stChatInput"] {
            border-color: #343b4b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
        create_conversation()

    conversations = st.session_state.conversations

    with st.sidebar:
        st.markdown("## ✦ Agentic Chat")
        st.caption("A multi-agent assistant powered by Ollama")
        if st.button("＋  New conversation", use_container_width=True):
            create_conversation()
            st.rerun()

        st.markdown("#### Recent conversations")
        conversation_ids = list(conversations)
        for conversation_id in reversed(conversation_ids):
            conversation = conversations[conversation_id]
            is_current = (
                conversation_id == st.session_state.current_conversation_id
            )
            if st.button(
                f"{'● ' if is_current else ''}{conversation['title']}",
                key=f"conversation-{conversation_id}",
                use_container_width=True,
                type="secondary" if is_current else "tertiary",
            ):
                st.session_state.current_conversation_id = conversation_id
                st.rerun()

        st.divider()
        st.caption("Routes questions to coding, research, planning, or general agents.")

    conversation = conversations[st.session_state.current_conversation_id]

    if not conversation["messages"]:
        st.markdown(
            """
            <div class="hero">
                <div class="hero-mark">✦</div>
                <h1>What can I help with?</h1>
                <p>Ask a question and the right specialist agent will take it from here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        suggestions = [
            "Explain how Python decorators work",
            "Make a step-by-step project plan",
            "What is the IRCTC cancellation policy?",
        ]
        columns = st.columns(len(suggestions))
        for column, suggestion in zip(columns, suggestions):
            with column:
                if st.button(suggestion, key=f"suggestion-{suggestion}", use_container_width=True):
                    st.session_state.pending_prompt = suggestion
                    st.rerun()
    else:
        st.title(conversation["title"])
        for message in conversation["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    metadata = message.get("agent", "Assistant")
                    if message.get("metrics"):
                        metadata += f" · {message['metrics']}"
                    st.caption(metadata)

    if conversation.get("error"):
        st.error(f"Could not get a response: {conversation['error']}")

    prompt = st.chat_input("Message your agents…")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Agents are working…"):
                submit_message(prompt, conversation)
        st.rerun()

    pending_prompt = st.session_state.pop("pending_prompt", None)
    if pending_prompt:
        with st.spinner("Agents are working…"):
            submit_message(pending_prompt, conversation)
        st.rerun()


if __name__ == "__main__":
    main()
