import argparse
import json
import math
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "skincare_knowledge.json"
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
CHAT_MODEL_ALIAS = "qwen2.5-0.5b"


def load_documents(path: Path = DATA_PATH):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def find_relevant(query_embedding, doc_embeddings, top_k=3):
    scores = []
    for i, doc_embedding in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)
        scores.append((i, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]


def build_document_text(doc):
    return f"{doc['title']}. {doc['content']}"


def answer_question(query, documents, embedding_client, doc_embeddings, chat_client, top_k=1):
    query_response = embedding_client.generate_embedding(query)
    query_embedding = query_response.data[0].embedding

    results = find_relevant(query_embedding, doc_embeddings, top_k=top_k)
    retrieved_docs = [documents[i] for i, _ in results]

    context_parts = []
    for number, doc in enumerate(retrieved_docs, start=1):
        context_parts.append(
            f"[Source {number}] {doc['title']}\n"
            f"Information: {doc['content']}\n"
            f"Source organization: {doc['source_name']}\n"
            f"URL: {doc['source_url']}"
        )
    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are Skincare RAG Assistant, an educational skincare information assistant. "
                "Use ONLY facts explicitly stated in the provided context. Do not add background knowledge. "
                "If a detail is not in the context, omit it. Never invent mechanisms, bacteria, safety claims, combinations, or treatment advice. "
                "Answer in 2 to 4 short sentences. Do not diagnose or prescribe. "
                "If the context is insufficient, say: 'The knowledge base does not contain enough information to answer that.' "
                "Base the answer on Source 1 and mention [Source 1] once.\n\n"
                f"Context:\n{context}"
            ),
        },
        {"role": "user", "content": query},
    ]

    print("\nAnswer: ", end="", flush=True)
    for chunk in chat_client.complete_streaming_chat(messages):
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print("\n")

    print("Retrieved sources:")
    for number, ((_, score), doc) in enumerate(zip(results, retrieved_docs), start=1):
        print(f"  {number}. {doc['title']} ({doc['source_name']}) — similarity {score:.3f}")
        print(f"     {doc['source_url']}")
    print()


def run(top_k=1, one_question=None):
    documents = load_documents()
    document_texts = [build_document_text(doc) for doc in documents]

    print("Initializing Foundry Local...")
    config = Configuration(app_name="skincare_rag_assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = None
    chat_model = None

    try:
        print(f"Loading embedding model: {EMBEDDING_MODEL_ALIAS}")
        embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
        embedding_model.download(
            lambda p: print(f"\rEmbedding model download: {p:.1f}%", end="", flush=True)
        )
        print()
        embedding_model.load()
        embedding_client = embedding_model.get_embedding_client()

        print(f"Indexing {len(document_texts)} skincare knowledge entries...")
        response = embedding_client.generate_embeddings(document_texts)
        doc_embeddings = [item.embedding for item in response.data]

        print(f"Loading chat model: {CHAT_MODEL_ALIAS}")
        chat_model = manager.catalog.get_model(CHAT_MODEL_ALIAS)
        chat_model.download(
            lambda p: print(f"\rChat model download: {p:.1f}%", end="", flush=True)
        )
        print()
        chat_model.load()
        chat_client = chat_model.get_chat_client()

        print("\nSkincare RAG Assistant is ready.")
        print("It answers from a small curated knowledge base and shows the retrieved sources.")
        print("Educational use only — not a medical diagnosis tool.\n")

        if one_question:
            answer_question(
                one_question,
                documents,
                embedding_client,
                doc_embeddings,
                chat_client,
                top_k=top_k,
            )
            return

        print("Example questions:")
        print('  - What does salicylic acid do for acne?')
        print('  - What is niacinamide used for?')
        print('  - Which ingredients can help dry skin?')
        print('  - Why is sunscreen important?')
        print('  - What is azelaic acid used for?')
        print('\nType "quit" to exit.\n')

        while True:
            query = input("Question: ").strip()
            if not query or query.lower() in {"quit", "exit"}:
                break
            answer_question(
                query,
                documents,
                embedding_client,
                doc_embeddings,
                chat_client,
                top_k=top_k,
            )
    finally:
        if embedding_model is not None:
            try:
                embedding_model.unload()
            except Exception:
                pass
        if chat_model is not None:
            try:
                chat_model.unload()
            except Exception:
                pass
        print("Models unloaded. Done.")


def parse_args():
    parser = argparse.ArgumentParser(description="Local skincare RAG assistant powered by Foundry Local")
    parser.add_argument("--top-k", type=int, default=1, help="Number of knowledge entries to retrieve")
    parser.add_argument("--question", type=str, help="Ask one question and exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(top_k=args.top_k, one_question=args.question)
