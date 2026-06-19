"""One-shot helper: discover your Telegram chat_id and update .env.

How it works:
  1. Reads TELEGRAM_BOT_TOKEN from .env.
  2. Calls Telegram's getUpdates API for the bot.
  3. Prints every chat_id seen in recent messages.
  4. Offers to update .env with the most recent chat_id automatically.

To use:
  1. In Telegram, find your bot (t.me/<bot_name>) and SEND IT ANY MESSAGE (/start works).
  2. Run: python -m futbot.scalp.get_chat_id
  3. The script prints the chat_id and (optionally) writes it to .env.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from futbot.config import _ENV_FILE


def _read_token() -> tuple[str | None, Path]:
    env = Path(_ENV_FILE)
    if not env.is_file():
        return None, env
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("TELEGRAM_BOT_TOKEN="):
            return line.split("=", 1)[1].strip(), env
    return None, env


def _patch_env(env: Path, chat_id: int):
    """Replace TELEGRAM_CHAT_ID=<old> with the new value, preserving the
    rest of the file. Adds a new line if the key isn't present."""
    lines = env.read_text(encoding="utf-8").splitlines()
    found = False
    out = []
    for line in lines:
        if line.strip().startswith("TELEGRAM_CHAT_ID="):
            out.append(f"TELEGRAM_CHAT_ID={chat_id}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"TELEGRAM_CHAT_ID={chat_id}")
    env.write_text("\n".join(out) + "\n", encoding="utf-8")


async def main():
    token, env = _read_token()
    if not token:
        print(f"ERROR: TELEGRAM_BOT_TOKEN not found in {env}")
        print("Open .env in the repo root and check it has TELEGRAM_BOT_TOKEN=<token>")
        sys.exit(1)

    from telegram import Bot

    bot = Bot(token=token)
    try:
        me = await bot.get_me()
        print(f"Connected to bot: @{me.username} (id={me.id})")
    except Exception as e:
        print(f"ERROR: bot.get_me() failed — token may be wrong.\n  {type(e).__name__}: {e}")
        sys.exit(2)

    print()
    print("Fetching recent updates...")
    try:
        updates = await bot.get_updates(timeout=5)
    except Exception as e:
        print(f"ERROR: get_updates failed: {e}")
        sys.exit(3)

    if not updates:
        print()
        print("=" * 60)
        print("Нет сообщений в очереди обновлений.")
        print()
        print("ЧТО ДЕЛАТЬ:")
        print(f"  1. Открой Telegram и найди бота: @{me.username}")
        print("  2. Нажми START или отправь ему любое сообщение (например /start)")
        print("  3. Запусти этот скрипт снова: python -m futbot.scalp.get_chat_id")
        print("=" * 60)
        sys.exit(0)

    # Collect unique chat_ids with most recent first
    seen: dict[int, str] = {}
    for u in updates:
        m = u.message or u.edited_message or u.channel_post
        if not m:
            continue
        cid = m.chat.id
        name = m.chat.full_name or m.chat.title or m.chat.username or str(cid)
        seen[cid] = name

    print()
    print(f"Найдено chat_id(ов): {len(seen)}")
    for cid, name in seen.items():
        print(f"  • {cid}  ({name})")

    if not seen:
        print("Странно — обновления есть, но без message объекта.  Попробуй ещё раз.")
        sys.exit(0)

    # Pick most recently active chat
    latest_cid = list(seen.keys())[-1]
    print()
    print(f"Самый недавний chat_id: {latest_cid}")
    print(f"Прописать его в {env}? [y/N]: ", end="", flush=True)
    answer = sys.stdin.readline().strip().lower()
    if answer in ("y", "yes", "д", "да"):
        _patch_env(env, latest_cid)
        print(f"  → записал TELEGRAM_CHAT_ID={latest_cid} в {env}")
        print(f"  → перезапусти бота, чтобы он подхватил")
    else:
        print(f"  Отменено.  Если хочешь — пропиши вручную:")
        print(f"      TELEGRAM_CHAT_ID={latest_cid}")


if __name__ == "__main__":
    asyncio.run(main())
