import asyncio
import sys
import os
from pathlib import Path

from dotenv import load_dotenv
from clients.llama_client import LlamaClient
from agent import run_agent_async

# Загружаем переменные из .env (ищем от корня проекта)
load_dotenv(Path(__file__).parent.parent / ".env")

MODEL    = os.getenv("LLAMA_MODEL",    "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")
BASE_URL = os.getenv("LLAMA_BASE_URL", "http://localhost:8080/v1")


def print_banner():
    print("=" * 60)
    print("My Generator v2: Native Tool Calls (llama-server)")
    print("=" * 60)
    print("Commands:")
    print("  quit  - exit")
    print("  clear - reset conversation")
    print("=" * 60)
    print()


async def main():
    print_banner()

    client = LlamaClient(model=MODEL, base_url=BASE_URL)

    conversation_history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            conversation_history.clear()
            print("Conversation cleared.\n")
            continue

        print("\nAgent: ", end="", flush=True)

        try:
            await run_agent_async(
                client=client,
                user_message=user_input,
                conversation_history=conversation_history,
            )
        except Exception as e:
            print(f"\n[AGENT ERROR] {e}", file=sys.stderr)

        print()  # пустая строка после ответа агента


if __name__ == "__main__":
    asyncio.run(main())
