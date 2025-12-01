#!/usr/bin/env python
"""Quick test of RAG API endpoints."""

import json

import requests

API_BASE = "http://localhost:8000"
TIMEOUT = 30  # seconds

print("Testing RAG API Server...")
print("=" * 60)

# Test 1: Root endpoint
print("\n1. Testing root endpoint...")
try:
    response = requests.get(f"{API_BASE}/", timeout=TIMEOUT)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Health endpoint
print("\n2. Testing health endpoint...")
try:
    response = requests.get(f"{API_BASE}/health", timeout=TIMEOUT)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.json()}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Stats endpoint
print("\n3. Testing stats endpoint...")
try:
    response = requests.get(f"{API_BASE}/stats", timeout=TIMEOUT)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.json()}")
except Exception as e:
    print(f"Error: {e}")

# Test 4: Query endpoint
print("\n4. Testing query endpoint...")
try:
    query_data = {"query": "What is RSI mean reversion?", "top_k": 3}
    response = requests.post(f"{API_BASE}/api/v1/query", json=query_data, timeout=TIMEOUT)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Query: {result['query']}")
        print(f"Query Type: {result['query_type']}")
        print(f"Results: {len(result['results'])}")
        # Handle both field names for execution time
        exec_time = result.get("execution_time_ms") or result.get("latency_ms", 0)
        print(f"Execution Time: {exec_time:.0f}ms")
        if result["results"]:
            print("\nFirst result:")
            print(f"  Score: {result['results'][0]['score']:.4f}")
            print(f"  Content: {result['results'][0]['content'][:100]}...")
    else:
        print(f"Error: {response.json()}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Testing complete!")
