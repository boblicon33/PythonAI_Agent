import asyncio
import sys
import os
import re

import feedparser
from fastmcp import FastMCP

# --- ЛЕНИВАЯ ИНИЦИАЛИЗАЦИЯ RAG ---
# HuggingFaceEmbeddings скачивает ~80MB модель при первом обращении.
# Чтобы MCP-сервер стартовал без сети, инициализацию отложили до первого
# вызова search_knowledge / add_to_knowledge.

_embeddings = None
_vectorstore = None
_init_error = None


def _ensure_vectorstore():
    global _embeddings, _vectorstore, _init_error

    if _vectorstore is not None:
        return _vectorstore
    if _init_error is not None:
        raise _init_error

    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vectorstore = Chroma(
            embedding_function=_embeddings,
            persist_directory="./chroma_db",
        )
        print("[MCP] RAG (ChromaDB + all-MiniLM-L6-v2) инициализирован успешно.", file=sys.stderr)
        return _vectorstore
    except Exception as e:
        _init_error = e
        raise


mcp = FastMCP("NewsDigestTools")


@mcp.tool()
def fetch_rss(url: str) -> str:
    """Fetches articles from an RSS feed and returns them as a text list."""
    try:
        feed = feedparser.parse(url)
        if feed.bozo:
            return f"Error: Failed to parse RSS feed. {feed.bozo_exception}"
        articles = []
        for entry in feed.entries[:10]:
            title = getattr(entry, "title", "No title")
            link = getattr(entry, "link", "No link")
            content = ""
            if hasattr(entry, "content") and entry.content:
                content = entry.content[0].value
            elif hasattr(entry, "summary"):
                content = entry.summary
            articles.append(f"TITLE: {title} | LINK: {link} | CONTENT: {content}")
        return "\n---\n".join(articles) if articles else "No articles found."
    except Exception as e:
        return f"Error fetching RSS: {str(e)}"


@mcp.tool()
def clean_html(html_text: str) -> str:
    """Removes HTML tags and extra whitespace from a text string."""
    text = re.sub(r'<script[\s\S]*?</script>', '', html_text, flags=re.IGNORECASE)
    text = re.sub(r'<style[\s\S]*?</style>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
    text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"')
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:300]


@mcp.tool()
def save_digest(text: str) -> str:
    """Saves the final digest text to a local file."""
    try:
        with open("digest_output.txt", "w", encoding="utf-8") as f:
            f.write(text)
        return "Digest saved successfully to digest_output.txt"
    except Exception as e:
        return f"Error saving digest: {str(e)}"


@mcp.tool()
def search_knowledge(query: str) -> str:
    """Searches the local knowledge base for similar past news.
    Use this to avoid duplicating news or to get context on a topic.
    """
    try:
        vs = _ensure_vectorstore()
        docs = vs.similarity_search(query, k=2)
        if not docs:
            return "В базе знаний пока нет информации по этой теме."

        results = []
        for i, doc in enumerate(docs):
            results.append(f"Фрагмент {i + 1}: {doc.page_content}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


@mcp.tool()
def add_to_knowledge(text: str, metadata: str = "") -> str:
    """Adds a new text fragment to the knowledge base."""
    try:
        vs = _ensure_vectorstore()
        vs.add_texts(texts=[text], metadatas=[{"source": metadata}])
        return "Успешно добавлено в базу знаний."
    except Exception as e:
        return f"Error adding to knowledge base: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")