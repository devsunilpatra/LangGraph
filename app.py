import streamlit as st
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered",  
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Sora:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0f1117;
}

/* ── Header ── */
.header-block {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border: 1px solid #2a3550;
    border-radius: 16px;
    padding: 28px 32px 20px;
    margin-bottom: 24px;
}
.header-block h1 {
    font-family: 'Sora', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 4px;
    letter-spacing: -0.3px;
}
.header-block p {
    color: #64748b;
    font-size: 0.88rem;
    margin: 0;
}

/* ── Programme badge ── */
.badge {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-top: 10px;
    letter-spacing: 0.3px;
}

/* ── Chat messages ── */
.msg-user {
    background: #1e293b;
    border: 1px solid #2d3f55;
    border-radius: 14px 14px 4px 14px;
    padding: 12px 16px;
    margin: 8px 0 8px auto;
    max-width: 78%;
    color: #e2e8f0;
    font-size: 0.9rem;
    line-height: 1.55;
    width: fit-content;
    margin-left: auto;
}
.msg-ai {
    background: #131b2e;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #3b82f6;
    border-radius: 4px 14px 14px 14px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 84%;
    color: #cbd5e1;
    font-size: 0.9rem;
    line-height: 1.6;
}
.msg-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
    color: #475569;
}
.msg-label.ai { color: #3b82f6; }
.msg-label.user { text-align: right; color: #475569; }

/* ── Tag chip ── */
.tag {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 10px;
    margin-top: 6px;
    letter-spacing: 0.3px;
}
.tag-academic { background: #14532d; color: #4ade80; }
.tag-fee      { background: #431407; color: #fb923c; }
.tag-general  { background: #312e81; color: #a5b4fc; }

/* ── Chat container ── */
.chat-area {
    max-height: 480px;
    overflow-y: auto;
    padding: 4px 2px;
}

/* ── Selectbox & input overrides ── */
.stSelectbox > div > div {
    background: #1a1f2e;
    border: 1px solid #2a3550;
    border-radius: 10px;
    color: #e2e8f0;
}
.stTextInput > div > div > input {
    background: #1a1f2e;
    border: 1px solid #2a3550;
    border-radius: 10px;
    color: #e2e8f0;
    padding: 10px 14px;
}
.stTextInput > div > div > input:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15);
}

/* ── Button ── */
.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 0.88rem;
    width: 100%;
    transition: background 0.15s;
}
.stButton > button:hover {
    background: #1d4ed8;
}

/* ── Divider ── */
hr { border-color: #1e293b; }

/* ── Clear button ── */
.clear-btn > button {
    background: transparent !important;
    border: 1px solid #2a3550 !important;
    color: #475569 !important;
    font-size: 0.8rem !important;
    padding: 6px 16px !important;
}
.clear-btn > button:hover {
    border-color: #ef4444 !important;
    color: #ef4444 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Build RAG (cached so it only runs once) ────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base…")
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def build_retriever(pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        document = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(document)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    academic_retriever = build_retriever("academics_handbook.pdf")
    fee_retriever = build_retriever("fee_structure.pdf")
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)
    return academic_retriever, fee_retriever, llm


academic_retriever, fee_retriever, llm = load_resources()


# ── LangGraph setup (unchanged logic) ─────────────────────────────────────────
class State(TypedDict):
    programm: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


def classifier_node(state: State) -> dict:
    last_message = state["messages"][-1].content
    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )
    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"
    return {"query_type": category}


def academic_rag_node(state: State) -> dict:
    query = state["messages"][-1].content
    docs = academic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def fee_rag_node(state: State) -> dict:
    query = state["messages"][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def general_node(state: State) -> dict:
    return {"retrieved_context": "NO RETRIEVAL NEEDED"}


def response_node(state: State) -> dict:
    query = state["messages"][-1].content
    programm = state.get("programm", "Unknown")
    context = state["retrieved_context"]
    if context == "NO RETRIEVAL NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programm} student."
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programm} student."
            f"Use the following context from the official college documents to answer"
            f"the question accurately. If the context mentions specific figures for "
            f"different programms, highlight the one relevant to {programm} "
            f"Context:\n{context}\n\n"
            f"question:\n{query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )
    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}


def route_query(state: State):
    if state["query_type"] == "academic":
        return "academic_rag"
    elif state["query_type"] == "fee":
        return "fee_rag"
    else:
        return "general"


@st.cache_resource
def build_graph():
    graph = StateGraph(State)
    graph.add_node("classifier", classifier_node)
    graph.add_node("academic_rag", academic_rag_node)
    graph.add_node("fee_rag", fee_rag_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)
    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", route_query)
    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")
    graph.add_edge("response", END)
    return graph.compile()


app = build_graph()


# ── Session state defaults ─────────────────────────────────────────────────────
if "programme" not in st.session_state:
    st.session_state.programme = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role", "content", "tag"}


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
    <h1>🎓 College Assistant</h1>
    <p>Ask anything about academics, fees, or general queries — I've got you covered.</p>
    {}
</div>
""".format(
    f'<span class="badge">📚 {st.session_state.programme}</span>'
    if st.session_state.programme else ""
), unsafe_allow_html=True)


# ── Programme selection ────────────────────────────────────────────────────────
if st.session_state.programme is None:
    st.markdown("#### Select your programme to get started")
    col1, col2 = st.columns([3, 1])
    with col1:
        choice = st.selectbox(
            "Programme",
            ["BCA", "BBA", "B.Com (H)"],
            label_visibility="collapsed",
        )
    with col2:
        if st.button("Continue →"):
            st.session_state.programme = choice
            st.rerun()
    st.stop()


# ── Chat history display ───────────────────────────────────────────────────────
tag_classes = {
    "academic": ("tag-academic", "Academic"),
    "fee": ("tag-fee", "Fee"),
    "general": ("tag-general", "General"),
}

if st.session_state.chat_history:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-label user">You</div>'
                f'<div class="msg-user">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            tag_html = ""
            if msg.get("tag") and msg["tag"] in tag_classes:
                cls, label = tag_classes[msg["tag"]]
                tag_html = f'<br><span class="tag {cls}">{label}</span>'
            st.markdown(
                f'<div class="msg-label ai">Assistant</div>'
                f'<div class="msg-ai">{msg["content"]}{tag_html}</div>',
                unsafe_allow_html=True,
            )
else:
    st.markdown(
        '<p style="color:#334155;font-size:0.88rem;text-align:center;padding:32px 0;">'
        'No messages yet. Ask your first question below.</p>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Input area ────────────────────────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Message",
            placeholder="Ask about attendance, fees, exams…",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    with st.spinner("Thinking…"):
        result = app.invoke({
            "programm": st.session_state.programme,
            "messages": [("human", user_input.strip())],
        })
        answer = result["messages"][-1].content
        query_type = result.get("query_type", "general")

    st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
    st.session_state.chat_history.append({"role": "ai", "content": answer, "tag": query_type})
    st.rerun()

# ── Footer controls ───────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
col_a, col_b, col_c = st.columns([3, 1.2, 1.2])
with col_b:
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
with col_c:
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("Switch programme"):
        st.session_state.programme = None
        st.session_state.chat_history = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)