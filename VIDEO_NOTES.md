# 2–3 Minute Video Notes

## 1. Problem and idea (20–30 sec)
I built a local RAG-based skincare ingredient assistant. Instead of asking a general chatbot to answer from everything it may know, my application first searches a small skincare knowledge base and then asks a local language model to answer using the retrieved information.

## 2. How it works (45–60 sec)
The knowledge base is stored as JSON entries with a title, information, source organization, and source URL. Foundry Local runs an embedding model that converts both the skincare entries and the user's question into vectors. I calculate cosine similarity to find the most relevant entries. Those entries become the context for a local chat model, which generates the final answer. The application also prints the retrieved sources.

## 3. Demo (30–45 sec)
Ask two questions, for example:
- What does salicylic acid do for acne?
- Which ingredients can help dry skin?
Show that the answer changes according to the retrieved documents and that source URLs are displayed.

## 4. What I learned (30–40 sec)
I learned the basic RAG pipeline, how embeddings and similarity search work, how to use Foundry Local models from Python, and why grounding an AI response in selected documents can make an application more controlled and traceable.
