import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI()

EXTRACTION_PROMPT = """You are a document extraction expert.
Extract all text, tables, key-value pairs, and structured content from this document image.
Preserve table structure using markdown. Preserve all numbers, dates, and named fields exactly.
Return only the extracted content — no commentary."""


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_from_image(image_path: str) -> str:
    b64 = _encode_image(image_path)
    suffix = Path(image_path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content.strip()


def extract_from_bytes(image_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content.strip()
