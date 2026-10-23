import asyncio
import json
import os
import uuid
from typing import TypedDict, Annotated, List, Dict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from a2a.client import create_client
from a2a.types import SendMessageRequest, Message, Part, Role


LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openai.bothub.chat/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")
A2A_URL = os.environ.get("A2A_URL", "http://localhost:8000")
CHECKPOINT_DB = os.environ.get("CHECKPOINT_DB", "./checkpoints.sqlite")


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    raw_articles: List[Dict]
    relevant_articles: List[Dict]
    final_digest: str
    iterations: int
    rss_source: str
    next_node: str


def mcp_text(result) -> str:
    """Extract text parts from an MCP CallToolResult."""
    parts = []
    for item in getattr(result, "content", []) or []:
        if getattr(item, "type", "") == "text":
            parts.append(item.text)
    return "\n".join(parts)


async def main():
    server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0.1,
        max_tokens=1000
    )

    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as memory:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)

                # ==========================================
                # УЗЕЛ 1: СБОРЩИК (Collector)
                # ==========================================
                async def collector_node(state: GraphState):
                    print(f"\n>>> [NODE: COLLECTOR] Сбор статей (попытка {state.get('iterations', 0) + 1})...")

                    source = state.get("rss_source", "https://habr.com/ru/rss/hubs/artificial_intelligence/articles/")

                    collector_agent = create_react_agent(llm, tools)
                    prompt = f"""You are a data collector.
1. Use fetch_rss to get articles from {source}
2. Take the first 3 articles.
3. For each article, use clean_html to clean the content.
Just perform these actions."""

                    response = await collector_agent.ainvoke({"messages": [HumanMessage(content=prompt)]})

                    articles = []
                    for msg in response["messages"]:
                        if isinstance(msg, ToolMessage) and msg.name == "clean_html":
                            text_content = msg.content[0]["text"] if isinstance(msg.content, list) else msg.content
                            articles.append({"title": "Статья", "link": "Ссылка", "text": text_content})

                    print(f"    Собрано статей: {len(articles)}")
                    return {
                        "raw_articles": articles,
                        "iterations": state.get("iterations", 0) + 1
                    }

                # ==========================================
                # УЗЕЛ 2: АНАЛИТИК (Analyst)
                # ==========================================
                async def analyst_node(state: GraphState):
                    print("\n>>> [NODE: ANALYST] Оценка релевантности с использованием RAG...")
                    articles = state["raw_articles"]
                    relevant = []

                    analyst_agent = create_react_agent(llm, tools)

                    for i, art in enumerate(articles):
                        print(f"  Анализ статьи {i + 1}...")

                        prompt = f"""You are an AI analyst. Your task:
1. Use the search_knowledge tool to check if this article is already in the knowledge base.
2. If the text is already in the base (duplicate), return {{"score": 0, "reasoning": "Duplicate"}}.
3. If the article is new, rate its relevance on a scale from 0 to 10.

Article: {art['text']}

Return ONLY JSON: {{"score": number, "reasoning": "why"}}"""

                        response = await analyst_agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
                        raw_text = response["messages"][-1].content

                        try:
                            res_json = json.loads(raw_text[raw_text.find('{'):raw_text.rfind('}') + 1])
                            score = res_json.get("score", 0)
                        except Exception:
                            score = 0

                        print(f"    Оценка: {score}")

                        if score >= 7:
                            art['score'] = score
                            relevant.append(art)

                    print(f"    Отобрано релевантных статей: {len(relevant)}")
                    return {"relevant_articles": relevant}

                # ==========================================
                # УЗЕЛ 3: СУПЕРВИЗОР (Supervisor)
                # ==========================================
                async def supervisor_node(state: GraphState):
                    print("\n>>> [NODE: SUPERVISOR] Принятие решения...")

                    relevant_count = len(state.get("relevant_articles", []))
                    iterations = state.get("iterations", 0)

                    print(f"    Релевантных статей: {relevant_count}, Итерация: {iterations}")

                    if relevant_count >= 1:
                        print("    -> Решение: статей достаточно, идем к Писателю.")
                        return {"next_node": "writer"}

                    if iterations >= 2:
                        print("    -> Решение: попытки исчерпаны, идем к Писателю (пустой дайджест).")
                        return {"next_node": "writer"}

                    print("    -> Решение: статей нет, просим Сборщика попробовать другой источник.")
                    new_source = "https://www.ixbt.com/export/news.rss" if iterations == 1 else "https://techcrunch.com/category/artificial-intelligence/feed/"

                    return {
                        "next_node": "collector",
                        "rss_source": new_source
                    }

                def route_supervisor(state: GraphState) -> str:
                    return state.get("next_node", "writer")

                # ==========================================
                # УЗЕЛ 4: ПИСАТЕЛЬ (Writer) + RAG + A2A
                # ==========================================
                async def writer_node(state: GraphState):
                    print("\n>>> [NODE: WRITER] Генерация дайджеста...")
                    articles = state["relevant_articles"]

                    if not articles:
                        return {"final_digest": "Релевантных новостей за сегодня не найдено."}

                    # RAG: извлекаем похожие прошлые новости, чтобы не дублировать
                    rag_context = ""
                    try:
                        rag_result = await session.call_tool("search_knowledge", {"query": "AI news digest"})
                        rag_context = mcp_text(rag_result)
                    except Exception as e:
                        print(f"    [RAG] Не удалось получить контекст: {e}")

                    context = "\n\n".join(
                        [f"[{a.get('title', 'Без заголовка')}]({a.get('link', '')})\n{a['text']}" for a in articles]
                    )
                    prompt = f"""You are a news digest editor.
Use the knowledge-base context below to avoid repeating already known news.

Knowledge-base context:
{rag_context or "None"}

Write a brief summary (3-4 sentences) based on these articles:

{context}"""

                    response = await llm.ainvoke([HumanMessage(content=prompt)])
                    digest = response.content

                    # Сохраняем статьи в базу знаний (детерминированно, напрямую через MCP)
                    print("    Сохранение статей в базу знаний...")
                    for art in articles:
                        metadata = f"{art.get('title', '')} | {art.get('link', '')}"
                        try:
                            await session.call_tool("add_to_knowledge", {"text": art["text"], "metadata": metadata})
                        except Exception as e:
                            print(f"    [KNOWLEDGE] Ошибка сохранения: {e}")

                    # --- ОТПРАВКА ПО A2A ---
                    print("    Отправка дайджеста по протоколу A2A...")
                    try:
                        client = await create_client(A2A_URL)

                        msg = Message(
                            role=Role.ROLE_USER,
                            parts=[Part(text=digest)],
                            message_id=str(uuid.uuid4())
                        )

                        request = SendMessageRequest(message=msg)

                        print("    [A2A] Ожидание ответа от сервера...")
                        async for response in client.send_message(request):
                            print(f"    [A2A] Ответ сервера: {response}")

                    except Exception as e:
                        print(f"    [A2A] Ошибка отправки: {e}")

                    print("    Дайджест сформирован.")
                    return {"final_digest": digest}

                # ==========================================
                # СБОРКА ГРАФА
                # ==========================================
                workflow = StateGraph(GraphState)

                workflow.add_node("collector", collector_node)
                workflow.add_node("analyst", analyst_node)
                workflow.add_node("supervisor", supervisor_node)
                workflow.add_node("writer", writer_node)

                workflow.set_entry_point("collector")
                workflow.add_edge("collector", "analyst")
                workflow.add_edge("analyst", "supervisor")

                workflow.add_conditional_edges(
                    "supervisor",
                    route_supervisor,
                    {
                        "writer": "writer",
                        "collector": "collector",
                    }
                )

                workflow.add_edge("writer", END)

                graph = workflow.compile(checkpointer=memory)

                # ==========================================
                # ЗАПУСК
                # ==========================================
                config = {"configurable": {"thread_id": "digest_orchestrated"}}
                print("🚀 ЗАПУСК МУЛЬТИАГЕНТНОЙ СИСТЕМЫ С ОРКЕСТРАЦИЕЙ...")

                final_state = await graph.ainvoke(
                    {"messages": [HumanMessage(content="Сделай дайджест новостей об ИИ.")]},
                    config=config
                )

                print("\n" + "=" * 50)
                print("📰 ФИНАЛЬНЫЙ ДАЙДЖЕСТ:")
                print(final_state["final_digest"])


if __name__ == "__main__":
    asyncio.run(main())