import asyncio
import websockets
import json

async def test_connection():
    import sys
    import os

    # Look for token in environment variable or command line arguments
    token = os.environ.get("ZEROCLAW_TOKEN")
    if not token and len(sys.argv) > 1:
        token = sys.argv[1]
    if not token:
        token = input("Please enter your ZeroClaw token (e.g., zc_...): ").strip()
        
    if not token:
        print("Error: Token cannot be empty. Exiting.")
        return

    uri = f'ws://host.docker.internal:42617/ws/avatar?token={token}'
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as ws:
            print("Connected successfully!")
            
            # Send a test ping or message
            test_msg = { # type: ignore
                "type": "chat",
                "messages": [{"role": "user", "content": "Hello! Are you there?"}]
            }
            await ws.send(json.dumps(test_msg))
            print(f"Sent: {json.dumps(test_msg)}")
            
            # Wait for a response chunks
            response = await ws.recv()
            print(f"Received: {response}")
            
    except Exception as e:
        print(f"Connection failed: {e}")

asyncio.run(test_connection())
