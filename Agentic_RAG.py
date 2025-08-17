"""
IITB Q&ASmartbot | Course Project
EE782 – Advanced Topics in Machine Learning (Sep'23–Nov'23)

End‑to‑end reference implementation that matches the resume bullets:
• Scrape 20+ IITB websites using BeautifulSoup/Requests to build a domain‑specific knowledge base
• Agentic RAG chatbot using LangGraph + ChromaDB with dynamic routing for accurate responses
• LLaMA‑2‑13B‑GPTQ model integration with LangSmith tracing, lightweight output moderation

NOTE: This is a runnable template. You’ll need to install deps and provide your own URL list
and API keys where noted. Model weights are not bundled; see instructions in README block.
"""
from __future__ import annotations

import os
import re
import sys
import time
import json
import math
import argparse
import dataclasses
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# --- Optional GPU / model imports (safe to import lazily in code paths that use them) ---
# transformers, accelerate, optimum, auto-gptq are commonly used with GPTQ models
# We'll import inside LLM wrapper to avoid import errors on machines without these packages.

# --- Vector store / embeddings ---
import chromadb
from chromadb.utils import embedding_functions

# --- LangGraph / LangChain style plumbing ---
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
except Exception:
    StateGraph = None
    MemorySaver = None
    END = "__END__"

# --- LangSmith (tracing/eval) ---
try:
    from langsmith import traceable
except Exception:
    def traceable(fn=None, **_):
        # No‑op decorator fallback if langsmith not installed
        if fn is None:
            return lambda f: f
        return fn

# ------------------------------
# Config
# ------------------------------

IITB_SEED_SITES = [
    # Replace/extend to reach 20+ domains (departments, labs, events, policies, course pages)
    "https://www.iitb.ac.in/",
    "https://www.ee.iitb.ac.in/",
    "https://www.cse.iitb.ac.in/",
    "https://www.iitb.ac.in/academics",
    "https://www.iitb.ac.in/en/education/academic-divisions",
    "https://www.iitb.ac.in/en/campuslife",
    "https://www.iitb.ac.in/en/research-development",
    "https://moodle.iitb.ac.in/",
    "https://www.iitb.ac.in/en/convocation",
    "https://www.iitb.ac.in/en/research-highlights",
    "https://www.ee.iitb.ac.in/web/academics/courses",
    "https://www.ee.iitb.ac.in/web/people/faculty",
    "https://www.cse.iitb.ac.in/page14",
    "https://www.iitb.ac.in/en/placement-cell",
    "https://www1.iitb.ac.in/newacadhome/toacadcalender.jsp",
    "https://portal.iitb.ac.in/",
    "https://gymkhana.iitb.ac.in/",
    "https://www.iitb.ac.in/en/covid-19-updates",
    "https://www.iitb.ac.in/en/bhavan",
    "https://www.iitb.ac.in/en/hostel",
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_iitb")
DOCS_PATH = os.path.join(DATA_DIR, "docs.jsonl")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval routing thresholds
MIN_RELEVANCE = 0.35  # cosine sim lower bound to trust RAG context
TOP_K = 5

# ------------------------------
# Utilities
# ------------------------------

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)


def clean_text(txt: str) -> str:
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"\u00A0", " ", txt)
    return txt.strip()


def is_same_domain(seed: str, link: str) -> bool:
    try:
        s = urlparse(seed).netloc
        l = urlparse(link).netloc
        return l.endswith(s.split(":")[0].split("@")[0]) or s.endswith(l)
    except Exception:
        return False


@dataclass
class Doc:
    id: str
    url: str
    title: str
    text: str


# ------------------------------
# Web Scraper (Requests + BeautifulSoup)
# ------------------------------

@traceable(name="crawl_site")
def crawl(seed_url: str, max_pages: int = 50, timeout: int = 10) -> List[Doc]:
    visited = set()
    queue = [seed_url]
    docs: List[Doc] = []
    session = requests.Session()

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        try:
            resp = session.get(url, timeout=timeout, headers={"User-Agent": "iitb-qna-bot/1.0"})
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.get_text(strip=True) if soup.title else url
            # Kill nav/footers/scripts
            for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
                tag.decompose()
            text = clean_text(soup.get_text(" "))
            if len(text) < 200:  # skip tiny pages
                continue
            doc_id = f"{hash(url)}"
            docs.append(Doc(id=doc_id, url=url, title=title, text=text))

            # enqueue same-domain links
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"]).split("#")[0]
                if link.startswith("mailto:") or link.startswith("tel:"):
                    continue
                if is_same_domain(seed_url, link) and link not in visited and link not in queue:
                    queue.append(link)
        except Exception:
            continue
    return docs


def crawl_many(seeds: List[str], max_pages_per_site: int = 30) -> List[Doc]:
    all_docs: List[Doc] = []
    for s in seeds:
        all_docs.extend(crawl(s, max_pages=max_pages_per_site))
    # de‑dup by URL
    uniq: Dict[str, Doc] = {}
    for d in all_docs:
        if d.url not in uniq or len(d.text) > len(uniq[d.url].text):
            uniq[d.url] = d
    return list(uniq.values())


# ------------------------------
# Persist & Load
# ------------------------------

def save_jsonl(docs: List[Doc], path: str = DOCS_PATH):
    ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(dataclasses.asdict(d), ensure_ascii=False) + "\n")


def load_jsonl(path: str = DOCS_PATH) -> List[Doc]:
    out: List[Doc] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            out.append(Doc(**j))
    return out


# ------------------------------
# ChromaDB indexing & retrieval
# ------------------------------

def get_chroma(embedding_model: str = EMBED_MODEL):
    ensure_dirs()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    collection = client.get_or_create_collection(name="iitb_qna", embedding_function=embed_fn, metadata={"hnsw:space": "cosine"})
    return client, collection


def index_docs(docs: List[Doc]):
    _, col = get_chroma()
    # upsert in batches
    B = 64
    for i in range(0, len(docs), B):
        chunk = docs[i:i+B]
        col.upsert(
            ids=[d.id for d in chunk],
            metadatas=[{"url": d.url, "title": d.title} for d in chunk],
            documents=[d.text for d in chunk],
        )


def retrieve(query: str, top_k: int = TOP_K):
    _, col = get_chroma()
    res = col.query(query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances", "embeddings"])
    # Chroma returns distances; for cosine space, similarity = 1 - distance
    sims = [1 - d for d in (res.get("distances", [[]])[0] or [])]
    docs = (res.get("documents", [[]])[0] or [])
    metas = (res.get("metadatas", [[]])[0] or [])
    items = []
    for i in range(len(docs)):
        items.append({
            "text": docs[i],
            "meta": metas[i],
            "similarity": sims[i]
        })
    return items


# ------------------------------
# Lightweight Moderation (placeholder)
# ------------------------------

PROFANITY = re.compile(r"\b(fuck|shit|bitch|asshole|bastard)\b", re.I)


def moderate_output(text: str) -> Dict[str, Any]:
    flagged = bool(PROFANITY.search(text))
    safe_text = text
    if flagged:
        safe_text = re.sub(PROFANITY, "[redacted]", text)
    return {"flagged": flagged, "text": safe_text}


# ------------------------------
# LLM Wrapper (LLaMA‑2‑13B‑GPTQ)
# ------------------------------

class Llama2GPTQ:
    def __init__(self, model_repo: str = "TheBloke/Llama-2-13B-GPTQ", device_map: str = "auto", trust_remote_code: bool = True):
        self.model_repo = model_repo
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None

    def load(self):
        if self._model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_repo, use_fast=True, trust_remote_code=self.trust_remote_code)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_repo,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=True
        )

    @traceable(name="llm_generate")
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
        self.load()
        import torch
        input_ids = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**input_ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0.0, temperature=temperature)
        return self._tokenizer.decode(out[0], skip_special_tokens=True)


# ------------------------------
# Prompting
# ------------------------------

RAG_PROMPT = (
    "You are IITB Q&A Smartbot. Use the provided context snippets from IIT Bombay webpages to answer the question.\n"
    "If the answer isn't in the context, say you don't know and suggest the closest relevant IITB resource.\n\n"
    "# Question\n{question}\n\n# Context\n{context}\n\n# Answer:\n"
)


# ------------------------------
# LangGraph Agentic RAG with Dynamic Routing
# ------------------------------

@dataclass
class GraphState:
    question: str
    route: str = "unknown"  # one of {"rag", "direct", "fallback"}
    contexts: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    answer: Optional[str] = None
    moderated: Optional[Dict[str, Any]] = None


def make_router_node():
    @traceable(name="router")
    def router(state: GraphState) -> GraphState:
        hits = retrieve(state.question, top_k=TOP_K)
        state.contexts = hits
        best = max([h.get("similarity", 0.0) for h in hits], default=0.0)
        state.route = "rag" if best >= MIN_RELEVANCE else "direct"
        return state
    return router


def make_rag_node(llm: Llama2GPTQ):
    @traceable(name="rag_generate")
    def rag(state: GraphState) -> GraphState:
        ctx_lines = []
        for i, h in enumerate(state.contexts):
            meta = h.get("meta", {})
            ctx_lines.append(f"[{i+1}] {meta.get('title','(untitled)')} — {meta.get('url','')}\n{h.get('text','')[:1000]}")
        context_block = "\n\n".join(ctx_lines) if ctx_lines else "(no relevant context)"
        prompt = RAG_PROMPT.format(question=state.question, context=context_block)
        out = llm.generate(prompt)
        state.answer = out.split("# Answer:")[-1].strip() if "# Answer:" in out else out
        return state
    return rag


def make_direct_node(llm: Llama2GPTQ):
    @traceable(name="direct_generate")
    def direct(state: GraphState) -> GraphState:
        prompt = (
            "You are IITB Q&A Smartbot. Answer concisely based on general IITB knowledge. If unsure, say you don't know.\n\n"
            f"Question: {state.question}\nAnswer:"
        )
        state.answer = llm.generate(prompt)
        return state
    return direct


def make_moderation_node():
    @traceable(name="moderate")
    def moderate(state: GraphState) -> GraphState:
        state.moderated = moderate_output(state.answer or "")
        return state
    return moderate


# Build the graph

def build_graph(llm: Llama2GPTQ):
    if StateGraph is None:
        raise RuntimeError("langgraph is not installed. Please `pip install langgraph`.\n")
    workflow = StateGraph(GraphState)
    router = make_router_node()
    rag = make_rag_node(llm)
    direct = make_direct_node(llm)
    mod = make_moderation_node()

    workflow.add_node("router", router)
    workflow.add_node("rag", rag)
    workflow.add_node("direct", direct)
    workflow.add_node("moderate", mod)

    workflow.set_entry_point("router")

    # Conditional edges: dynamic routing based on retrieval similarity
    def route_decider(state: GraphState):
        return state.route

    workflow.add_conditional_edges(
        "router",
        route_decider,
        {
            "rag": "rag",
            "direct": "direct",
            # fallback could be added here (e.g., web search or tool use)
        },
    )

    # Both generation paths go to moderation
    workflow.add_edge("rag", "moderate")
    workflow.add_edge("direct", "moderate")

    # End after moderation
    workflow.add_edge("moderate", END)

    memory = MemorySaver() if MemorySaver is not None else None
    app = workflow.compile(checkpointer=memory)
    return app


# ------------------------------
# CLI / Orchestrator
# ------------------------------

@traceable(name="ask_bot")
def ask(question: str, app, **kwargs) -> Dict[str, Any]:
    state = GraphState(question=question)
    out = app.invoke(state)
    payload = dataclasses.asdict(out)
    return payload


README = r"""
# IITB Q&A Smartbot (Agentic RAG)

## 0) Environment
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install requests beautifulsoup4 chromadb sentence-transformers langgraph langsmith transformers accelerate auto-gptq
```

Set LangSmith (optional, for tracing/eval):
```bash
export LANGSMITH_API_KEY=...  # or set in your shell profile
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="IITB-QnA-Smartbot"
```

## 1) Crawl & Ingest (20+ IITB sites)
```bash
python iitb_qna_smartbot.py crawl --max-pages 30
python iitb_qna_smartbot.py ingest
```

> Edit `IITB_SEED_SITES` in the script to include the exact 20+ IITB websites you want.

## 2) Run Q&A
```bash
python iitb_qna_smartbot.py chat --q "When does the semester start for EE?"
```

## 3) Model Weights (LLaMA‑2‑13B‑GPTQ)
- This template expects a GPTQ quantized repo like `TheBloke/Llama-2-13B-GPTQ` on Hugging Face.
- Ensure you have access (license) and sufficient VRAM (>= 12‑16GB typically for 13B GPTQ). You can also swap to a smaller model.

## Notes
- Dynamic routing: if retrieval similarity < `MIN_RELEVANCE`, bot uses a direct answer path.
- Output moderation is a simple placeholder; replace with your policy (e.g., Azure/OpenAI/Guardrails/Detoxify).
- Add tools or fallback routes (e.g., live web) as needed.
"""


def main():
    parser = argparse.ArgumentParser(description="IITB Q&A Smartbot – Agentic RAG")
    sub = parser.add_subparsers(dest="cmd")

    p_crawl = sub.add_parser("crawl", help="crawl seed sites")
    p_crawl.add_argument("--max-pages", type=int, default=30)

    p_ing = sub.add_parser("ingest", help="index docs to Chroma")

    p_chat = sub.add_parser("chat", help="ask a question")
    p_chat.add_argument("--q", required=True)
    p_chat.add_argument("--model-repo", default="TheBloke/Llama-2-13B-GPTQ")

    p_readme = sub.add_parser("readme", help="print README")

    args = parser.parse_args()

    if args.cmd == "crawl":
        print("[crawl] Starting crawl…")
        docs = crawl_many(IITB_SEED_SITES, max_pages_per_site=args.max_pages)
        print(f"[crawl] Collected {len(docs)} pages. Saving to {DOCS_PATH}")
        save_jsonl(docs)
        return

    if args.cmd == "ingest":
        docs = load_jsonl()
        if not docs:
            print("No docs found. Run crawl first.")
            return
        print(f"[ingest] Indexing {len(docs)} docs to Chroma at {CHROMA_DIR}…")
        index_docs(docs)
        print("[ingest] Done.")
        return

    if args.cmd == "chat":
        llm = Llama2GPTQ(model_repo=args.model_repo)
        app = build_graph(llm)
        payload = ask(args.q, app)
        ans = payload.get("answer") or ""
        moderated = payload.get("moderated", {})
        if moderated.get("flagged"):
            print("[moderation] Output had policy‑flagged content; showing redacted answer.")
            ans = moderated.get("text", ans)
        print("\n=== Answer ===\n")
        print(ans)
        print("\n=== Route ===\n", payload.get("route"))
        if payload.get("contexts"):
            print("\n=== Top Contexts ===")
            for i, ctx in enumerate(payload["contexts"], 1):
                meta = ctx.get("meta", {})
                print(f"[{i}] sim={ctx.get('similarity',0):.3f} | {meta.get('title','')} | {meta.get('url','')}")
        return

    if args.cmd == "readme":
        print(README)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
