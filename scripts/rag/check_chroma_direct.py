from pathlib import Path

import chromadb


def check_db():
    db_path = Path("data/chromadb")
    print(f"Checking DB at {db_path.absolute()}")

    try:
        client = chromadb.PersistentClient(path=str(db_path))
        print("Client initialized")

        cols = client.list_collections()
        print(f"Collections found: {[c.name for c in cols]}")

        for col in cols:
            print(f"Collection: {col.name}")
            print(f"  Count: {col.count()}")
            print(f"  Peek: {col.peek(limit=1)}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    check_db()
