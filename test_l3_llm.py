import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from airs_v2.perception.l3_llm import L3LLMFallback

async def test_l3():
    from dotenv import load_dotenv
    load_dotenv("/home/VenuSai/DevOpsAgent/.env")

    # Force use of groq API key if available
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  Warning: GROQ_API_KEY not set. Test might fall back to stub mode.", file=sys.stderr)

    l3 = L3LLMFallback()
    
    mock_log = (
        "info: Grpc.AspNetCore.Server.ServerCallHandler[7]\n"
        "      Error status code 'FailedPrecondition' with detail 'Can't access cart storage. System.ApplicationException: Wasn't able to connect to redis\n"
        "         at cart.cartstore.ValkeyCartStore.EnsureRedisConnected() in /usr/src/app/src/cartstore/ValkeyCartStore.cs:line 101"
    )
    
    print("Testing L3 LLM Fallback parser...")
    print(f"Log input:\n{mock_log}\n")
    print("Waiting for LLM response (this tests the <think> stripping logic)...\n")
    
    result = await l3.classify(mock_log)
    
    print("=" * 60)
    print("LLM Classification Result:")
    print("=" * 60)
    print(f"Template Key  : {result.template_key}")
    print(f"Regex Pattern : {result.regex_pattern}")
    print(f"Description   : {result.description}")
    print(f"Confidence    : {result.confidence}")
    print(f"Source        : {result.source}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_l3())
