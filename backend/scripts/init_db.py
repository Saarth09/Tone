"""Create Tone database tables.

Usage (from backend/ with venv active):
  python -m scripts.init_db
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import DATABASE_URL, database_info, init_db  # noqa: E402


async def main() -> None:
    print(f"Connecting via {DATABASE_URL.split('://', 1)[0]}://…")
    await init_db()
    info = await database_info()
    print("Database ready:")
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
