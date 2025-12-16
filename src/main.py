import asyncio
import sys

from clients.ollama_client import OllamaClient
from agent import run_agent_async

MODEL = "deepseek-coder:6.7b"
    
HOST = "http://localhost:11434"

def print_banner():
    print("=" * 60)
    print("My Generator v1: Minimum Viable Coding Agent (Ollama)")
    print("=" * 60)
    print("Commands:")
    print("  quit  - exit")
    print("  clear - reset conversation")
    print("=" * 60)
    print()

async def main():
    print_banner()

    # Инициализация клиента Ollama
    client = OllamaClient(
        model=MODEL,
        host=HOST
    )

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
                conversation_history=conversation_history
            )
        except Exception as e:
            print(f"\n[AGENT ERROR] {e}", file=sys.stderr)

        print()  # пустая строка после ответа агента


if __name__ == "__main__":
    asyncio.run(main())