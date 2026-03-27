"""
ThyraX CDSS — Agent Test Script
Tests the /agent/chat endpoint with a complex medical query
that triggers both the VectorDB (RAG) and SQL history tools.
"""

import sys
import requests
import json

BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/agent/chat"

# ── Test Payload ──────────────────────────────────────────────
PAYLOAD = {
    "query": (
        "According to the ATA guidelines, what is the recommended management "
        "plan for a patient who is hypothyroid and has a palpable thyroid nodule? "
        "Please also retrieve the full clinical history for this patient so I can "
        "compare the current labs with their previous visits."
    ),
    "patient_id": 1,
}

DIVIDER = "═" * 70


def print_section(title: str, content: str, color_code: str = ""):
    reset = "\033[0m"
    print(f"\n{color_code}{'─' * 70}{reset}")
    print(f"{color_code}  {title}{reset}")
    print(f"{color_code}{'─' * 70}{reset}")
    print(content)


def run_test():
    print(f"\n\033[1;34m{DIVIDER}")
    print("  ThyraX CDSS — LangChain Agent Test")
    print(f"{DIVIDER}\033[0m")

    # 1. Health check
    print("\n\033[1;33m[1/3] Checking server health...\033[0m")
    try:
        health = requests.get(BASE_URL, timeout=5)
        health.raise_for_status()
        msg = health.json().get("message", "")
        print(f"  ✅ Server is running: {msg}")
    except requests.ConnectionError:
        print(f"  ❌ Cannot connect to server at {BASE_URL}")
        print("     Start the server first:")
        print('     cd "/mnt/work/thyroid api/thyroid_canser"')
        print("     uv run python -m uvicorn main:app --port 8000")
        sys.exit(1)
    except Exception as e:
        print(f"  ❌ Server error: {e}")
        sys.exit(1)

    # 2. Print the question
    print_section(
        "📋 [2/3] Sending Query to /agent/chat",
        f"  Patient ID : {PAYLOAD['patient_id']}\n"
        f"  Query      :\n\n    {PAYLOAD['query']}",
        "\033[1;36m",
    )

    # 3. Send request
    print(f"\n\033[1;33m[3/3] Waiting for LangChain Agent response...\033[0m")
    print("  (This may take 15-30s — agent is searching guidelines + patient history)\n")

    try:
        response = requests.post(
            ENDPOINT,
            json=PAYLOAD,
            timeout=120,  # Agent can take a while
        )
        response.raise_for_status()
    except requests.Timeout:
        print("  ❌ Request timed out after 120s.")
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"  ❌ HTTP error: {e}")
        print(f"  Response body: {response.text[:500]}")
        sys.exit(1)
    except Exception as e:
        print(f"  ❌ Request failed: {e}")
        sys.exit(1)

    data = response.json()

    # 4. Print results
    tools_used = data.get("tools_used", [])
    tools_str = ", ".join(tools_used) if tools_used else "none"

    print_section(
        "🔧 Tools Invoked by the Agent",
        f"  {tools_str}",
        "\033[1;35m",
    )

    print_section(
        "🩺 Agent Response",
        data.get("response", "No response"),
        "\033[1;32m",
    )

    print_section(
        "⚕️  CDSS Disclaimer",
        "  " + data.get("disclaimer", "").replace("⚕️", "").strip(),
        "\033[1;31m",
    )

    print(f"\n\033[1;34m{DIVIDER}")
    print(f"  Status  : {data.get('status', 'unknown').upper()}")
    print(f"  HTTP    : {response.status_code}")
    print(f"{DIVIDER}\033[0m\n")


if __name__ == "__main__":
    run_test()
