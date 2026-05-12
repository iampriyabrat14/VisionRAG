from openai import OpenAI
from app.embeddings.embedder import embed_text
from app.db.vector_store import search

client = OpenAI()

SYSTEM_PROMPT = """You are a document Q&A assistant.
Answer the user's question using ONLY the provided document excerpts.
If the answer is not in the excerpts, say "I could not find this information in the uploaded documents."
Always cite the source document and page number for each piece of information you use.
Be concise and factual."""


def answer_query(query: str, top_k: int = 5) -> dict:
    query_embedding = embed_text(query)
    results = search(query_embedding, top_k=top_k)

    if not results:
        return {
            "answer": "No documents have been uploaded yet.",
            "citations": [],
            "context_chunks": [],
        }

    context_blocks = []
    for i, r in enumerate(results, start=1):
        context_blocks.append(
            f"[Source {i}: {r['filename']}, page {r['page_num']}]\n{r['content']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Document excerpts:\n\n{context}\n\nQuestion: {query}",
            },
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()

    citations = [
        {"source": r["filename"], "page": r["page_num"], "score": round(r["score"], 4)}
        for r in results
    ]

    return {
        "answer": answer,
        "citations": citations,
        "context_chunks": [r["content"][:300] + "..." for r in results],
    }
