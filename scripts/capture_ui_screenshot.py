#!/usr/bin/env python3
"""Capture a high-fidelity, fully rendered screenshot of the Streamlit UI."""
import asyncio
import base64
import json
import subprocess
import time
import urllib.request
from pathlib import Path
from PIL import Image
import websockets

CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9222
URL = "http://localhost:8501"
OUT_PATHS = [
    Path("reports/figures/ui_dashboard.png"),
    Path("graphs/ui_dashboard.png"),
]

def start_chrome():
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        f"--remote-debugging-port={PORT}",
        f"--window-size=1200,900",
        "about:blank",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

async def capture():
    proc = start_chrome()
    try:
        # Wait for Chrome CDP to respond
        ws_url = None
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"http://localhost:{PORT}/json/list", timeout=1) as resp:
                    tabs = json.loads(resp.read().decode())
                    if tabs:
                        ws_url = tabs[0]["webSocketDebuggerUrl"]
                        break
            except Exception:
                pass
            await asyncio.sleep(0.3)

        if not ws_url:
            raise RuntimeError("Could not connect to Chrome DevTools Protocol")

        print(f"[*] Connected to Chrome CDP: {ws_url}")
        async with websockets.connect(ws_url, max_size=25 * 1024 * 1024) as ws:
            msg_id = 1

            async def send(method, params=None):
                nonlocal msg_id
                m = {"id": msg_id, "method": method, "params": params or {}}
                msg_id += 1
                await ws.send(json.dumps(m))
                while True:
                    res = json.loads(await ws.recv())
                    if res.get("id") == m["id"]:
                        return res.get("result", {})

            await send("Page.enable")
            await send("Emulation.setDeviceMetricsOverride", {
                "width": 1200,
                "height": 900,
                "deviceScaleFactor": 1,
                "mobile": False,
            })
            print(f"[*] Navigating to {URL}...")
            await send("Page.navigate", {"url": URL})

            # Wait 8 seconds for Streamlit's React components, KPIs, and Plotly to render
            print("[*] Waiting 8 seconds for Streamlit components to mount and render...")
            await asyncio.sleep(8)

            print("[*] Capturing full screenshot...")
            res = await send("Page.captureScreenshot", {"format": "png"})
            img_bytes = base64.b64decode(res["data"])

            for p in OUT_PATHS:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(img_bytes)
                im = Image.open(p)
                print(f"[+] Saved screenshot to {p} (size={im.size}, bytes={len(img_bytes):,})")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    asyncio.run(capture())
