import os
import sys

import chromadb

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))


def inspect_chroma():
    persist_path = os.path.join(os.getcwd(), "data", "chromadb")
    print(f"Inspecting ChromaDB at: {persist_path}")

    try:
        client = chromadb.PersistentClient(path=persist_path)
        collections = client.list_collections()

        print(f"Found {len(collections)} collections:")
        for col in collections:
            count = col.count()
            print(f"- Name: {col.name}, Count: {count}")
            if count > 0:
                print(f"  - Sample metadata: {col.peek(1)['metadatas'][0]}")

    except Exception as e:
        print(f"Error inspecting ChromaDB: {e}")


if __name__ == "__main__":
    inspect_chroma()
