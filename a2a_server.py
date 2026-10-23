import os

import uvicorn
from starlette.applications import Starlette
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.types import (
    AgentCard, AgentSkill, AgentCapabilities, AgentInterface,
    Part
)
from a2a.helpers import new_task_from_user_message
from a2a.server.tasks import TaskUpdater


A2A_INTERFACE_URL = os.environ.get("A2A_INTERFACE_URL", "http://localhost:8000/")


# 1. Исполнитель
class DigestReceiverExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        print(f"\n[A2A SERVER] Получена задача от агента!")
        user_input = context.get_user_input()
        print(f"[A2A SERVER] Текст дайджеста (первые 100 символов): {user_input[:100]}...")

        with open("a2a_digest_output.txt", "w", encoding="utf-8") as f:
            f.write(user_input)

        # Создаём TaskUpdater для управления задачей
        # context.current_task содержит задачу, если она уже была создана
        # Если нет — создаём новую из сообщения
        task = context.current_task or new_task_from_user_message(context.message)

        # Отправляем задачу как первый event (обязательное правило SDK)
        await event_queue.enqueue_event(task)

        # Создаём TaskUpdater и завершаем задачу с подтверждением
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        confirmation = updater.new_agent_message([
            Part(text="Дайджест успешно получен и сохранён сервером.")
        ])
        await updater.complete(message=confirmation)

        print("[A2A SERVER] Дайджест сохранен в a2a_digest_output.txt, подтверждение отправлено.")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


# 2. Карточка агента
agent_card = AgentCard(
    name="DigestReceiver",
    description="Receives news digests from the main agent",
    version="1.0.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=False),
    supported_interfaces=[
        AgentInterface(url=A2A_INTERFACE_URL, protocol_binding="JSONRPC", protocol_version="1.0")
    ],
    skills=[AgentSkill(
        id="receive_digest",
        name="Receive Digest",
        description="Receives digest text",
        tags=["digest"]
    )]
)


def create_app():
    executor = DigestReceiverExecutor()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    routes = []
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(create_jsonrpc_routes(handler, rpc_url='/'))

    return Starlette(routes=routes)


if __name__ == "__main__":
    app = create_app()
    print("🚀 A2A Сервер запущен на http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)