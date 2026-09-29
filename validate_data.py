import json
from pathlib import Path

path = Path(__file__).resolve().parent / "data" / "skincare_knowledge.json"
required_fields = {"title", "content", "source_name", "source_url"}

with path.open("r", encoding="utf-8") as f:
    docs = json.load(f)

assert isinstance(docs, list) and docs, "Knowledge base must be a non-empty list."

for index, doc in enumerate(docs, start=1):
    missing = required_fields - set(doc)
    assert not missing, f"Entry {index} is missing: {sorted(missing)}"
    assert doc["source_url"].startswith("https://"), f"Entry {index} has an invalid source URL."

print(f"Knowledge base OK: {len(docs)} entries validated.")
