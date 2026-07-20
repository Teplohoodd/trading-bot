"""Auto-tunnel for the futbot Mini App.

Starts a Cloudflare quick tunnel to the local webapp (port 8088), grabs the
rotating https://…trycloudflare.com URL it prints, and automatically:
  1. writes WEBAPP_URL=… into .env, and
  2. re-points the Telegram menu button at that URL (Bot API).

So after the tunnel URL rotates (it changes every restart), everything is
re-wired with zero manual steps.  The running bot needs no restart: /app reads
.env on each call and the menu button is set directly via the API here.

  python -m futbot.webapp.tunnel

Uses --protocol http2 (QUIC/UDP is often blocked and was timing out on this box).
"""

import json
import re
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ENV = ROOT / ".env"
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
CTX = ssl._create_unverified_context()


def update_env(url: str):
    lines = ENV.read_text(encoding="utf-8").splitlines() if ENV.exists() else []
    lines = [ln for ln in lines if not ln.strip().startswith("WEBAPP_URL=")]
    lines.append(f"WEBAPP_URL={url}")
    ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[tunnel] .env → WEBAPP_URL={url}", flush=True)


def set_menu_button(url: str):
    from futbot.orchestrator.config import OrchSettings
    s = OrchSettings()
    if not s.TELEGRAM_CHAT_ID:
        print("[tunnel] no TELEGRAM_CHAT_ID — menu button skipped"); return
    payload = {"chat_id": s.TELEGRAM_CHAT_ID,
               "menu_button": {"type": "web_app", "text": "📊 futbot",
                               "web_app": {"url": url}}}
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{s.TELEGRAM_BOT_TOKEN}/setChatMenuButton",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=15, context=CTX).read())
        print(f"[tunnel] menu button set: ok={r.get('ok')}", flush=True)
    except Exception as e:
        print(f"[tunnel] menu button FAILED: {e}", flush=True)


def _find_cloudflared() -> str | None:
    import os
    import shutil
    p = shutil.which("cloudflared")
    if p:
        return p
    for cand in (
        r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
        r"C:\Program Files\cloudflared\cloudflared.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\cloudflared.exe"),
    ):
        if Path(cand).exists():
            return cand
    return None


def main():
    exe = _find_cloudflared()
    if not exe:
        print("[tunnel] cloudflared not found.\n"
              "  winget install Cloudflare.cloudflared")
        return
    print("[tunnel] starting Cloudflare tunnel → http://localhost:8088 …", flush=True)
    proc = subprocess.Popen(
        [exe, "tunnel", "--protocol", "http2", "--url", "http://localhost:8088"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=1)
    current = None
    try:
        for line in proc.stdout:
            line = line.rstrip()
            m = URL_RE.search(line)
            if m and m.group(0) != current:
                current = m.group(0)
                print(f"\n[tunnel] ✅ URL: {current}", flush=True)
                update_env(current)
                set_menu_button(current)
                print("[tunnel] Готово — открой бота в Telegram (кнопка 📊 futbot "
                      "или /app). Оставь это окно открытым.\n", flush=True)
            elif line and ("ERR" in line or "connection" in line.lower()
                           or "Registered" in line):
                print(f"[tunnel] {line[-100:]}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        print("[tunnel] stopped.")


if __name__ == "__main__":
    main()
