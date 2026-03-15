# =============================================================================
# main.py — CLI entry point
# =============================================================================
# Usage:
#   python main.py                        # interactive mode (loop)
#   python main.py "your question here"   # single query mode

import asyncio
import sys

from orchestrator import run


BANNER = """
╔══════════════════════════════════════════════════╗
║           LOCAL REASONING ASSISTANT              ║
║   deepseek-r1:8b · 6 hats · ChromaDB memory     ║
╚══════════════════════════════════════════════════╝
Type your query and press Enter. Type 'exit' to quit.
"""


async def main():
    print(BANNER)

    # Single query mode (argument passed on the command line)
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        await run(query)
        return

    # Interactive loop mode
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        await run(query)


if __name__ == "__main__":
    asyncio.run(main())