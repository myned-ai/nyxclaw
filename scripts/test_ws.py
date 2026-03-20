"""Quick WebSocket integration test against nyxclaw.

Connects to ws://localhost:8081/ws, sends a text message,
and prints every response with timestamps so we can measure
time-to-first-response and see what the mobile app would receive.

Usage:
    python scripts/test_ws.py "What is the capital of France?"
"""

import asyncio
import json
import sys
import time

import websockets


async def main(prompt: str) -> None:
    uri = "ws://localhost:8081/ws"
    print(f"Connecting to {uri} ...")

    async with websockets.connect(uri) as ws:
        t0 = time.time()
        print(f"[{0:.3f}s] Connected\n")

        # Send text message (no audio, just text)
        msg = {"type": "text", "data": prompt}
        await ws.send(json.dumps(msg))
        print(f"[{time.time() - t0:.3f}s] Sent: {prompt!r}\n")

        first_response = True
        try:
            async for raw in ws:
                elapsed = time.time() - t0
                data = json.loads(raw)
                msg_type = data.get("type", "?")

                if first_response:
                    print(f"[{elapsed:.3f}s] *** FIRST RESPONSE (TTFR: {elapsed:.3f}s) ***")
                    first_response = False

                # Pretty-print based on type
                if msg_type == "sync_frame":
                    # These are high-frequency — just count them
                    bs = data.get("blendshapes")
                    has_audio = bool(data.get("audio"))
                    transcript = data.get("transcript", "")
                    extra = ""
                    if transcript:
                        extra = f' transcript="{transcript}"'
                    if bs:
                        extra += f" blendshapes={len(bs)} weights"
                    print(f"[{elapsed:.3f}s] sync_frame  audio={'yes' if has_audio else 'no'}{extra}")
                elif msg_type == "transcript_delta":
                    print(f"[{elapsed:.3f}s] transcript_delta: {data.get('text', '')!r}")
                elif msg_type == "rich_content":
                    content = data.get("content", "")
                    print(f"[{elapsed:.3f}s] rich_content ({len(content)} chars):")
                    for line in content.split("\n")[:5]:
                        print(f"             {line}")
                    if content.count("\n") > 5:
                        print(f"             ... ({content.count(chr(10))} lines total)")
                elif msg_type == "response_end":
                    print(f"[{elapsed:.3f}s] response_end")
                    break
                elif msg_type == "interrupted":
                    print(f"[{elapsed:.3f}s] interrupted")
                    break
                elif msg_type == "error":
                    print(f"[{elapsed:.3f}s] ERROR: {data.get('message', data)}")
                    break
                elif msg_type in ("connect.challenge", "connect.result"):
                    print(f"[{elapsed:.3f}s] auth: {msg_type}")
                else:
                    # Print any other message type fully
                    print(f"[{elapsed:.3f}s] {msg_type}: {json.dumps(data, indent=2)[:200]}")

        except websockets.exceptions.ConnectionClosed as e:
            print(f"\n[{time.time() - t0:.3f}s] Connection closed: {e}")

    print(f"\n[{time.time() - t0:.3f}s] Done")


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?"
    asyncio.run(main(prompt))
