# Local RAG-Based Skincare Ingredient Assistant

A small Retrieval-Augmented Generation (RAG) project built with **Microsoft Foundry Local**. The assistant runs AI models locally, retrieves relevant skincare information from a curated knowledge base, and generates an answer grounded in those retrieved sources.

## Project goal

General-purpose chatbots can answer skincare questions from broad model knowledge, but their answers are not necessarily grounded in a specific set of references. This project demonstrates a simple RAG pipeline that:

1. Stores a small skincare knowledge base.
2. Converts each knowledge entry into an embedding vector.
3. Converts the user's question into an embedding.
4. Uses cosine similarity to retrieve the most relevant entries.
5. Sends only the retrieved context to a local chat model.
6. Shows the retrieved source links after every answer.

The project is educational and **not a medical diagnosis or treatment tool**.

## Tech stack

- Python 3.11+
- Microsoft Foundry Local SDK
- `qwen3-embedding-0.6b` for embeddings
- `qwen2.5-0.5b` for local chat generation
- JSON knowledge base
- Cosine similarity retrieval

## Project structure

```text
skincare-rag-foundry-local/
├── data/
│   └── skincare_knowledge.json
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Create a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
py -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python main.py
```

The first run downloads the local embedding and chat models. Later runs use the local cache and should start faster.

You can also ask one question and exit:

```bash
python main.py --question "What does salicylic acid do for acne?"
```

## Example questions

- What does salicylic acid do for acne?
- What is niacinamide used for?
- Which ingredients can help dry skin?
- What is azelaic acid used for?
- Why is sunscreen important?
- What is the basic order of a skincare routine?

## How the RAG pipeline works

```text
User question
    ↓
Embedding model
    ↓
Question vector
    ↓
Cosine similarity search
    ↓
Top skincare knowledge entries
    ↓
Context + user question
    ↓
Local chat model
    ↓
Grounded answer + retrieved sources
```

## Knowledge sources

The starter knowledge base contains short paraphrased entries derived from educational material from:

- American Academy of Dermatology (AAD): https://www.aad.org/
- DermNet: https://dermnetnz.org/

Each entry stores its own source URL in `data/skincare_knowledge.json`.

## What I learned

This project demonstrates:

- What Retrieval-Augmented Generation (RAG) is.
- How text embeddings represent semantic meaning numerically.
- How cosine similarity can retrieve relevant information.
- How a chat model can be grounded using retrieved context.
- How local AI models can run without sending the knowledge base to a cloud model.
- How to organize a small AI project with a separate data layer and documented sources.

## Possible improvements

- Add more skincare ingredients and evidence sources.
- Add chunking for longer documents.
- Store embeddings so they do not need to be regenerated at every run.
- Add a graphical or web interface.
- Add automated evaluation questions.
- Add filters for ingredient category or skin concern.

## Disclaimer

This assistant is for educational purposes only. It does not diagnose skin conditions or replace advice from a qualified healthcare professional.

## Demo Videosu

Projenin çalışma mantığını ve öğrendiklerimi anlattığım kısa video:

[Demo Videosunu İzle](https://drive.google.com/file/d/1cudEiCzviWUqvwHjlNxnQBTqgw-f59dQ/view?usp=sharing)