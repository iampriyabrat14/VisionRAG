import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.vision.extractor import extract_from_bytes
from app.vision.pdf_to_image import pdf_to_images
from app.embeddings.embedder import embed_text
from app.db.vector_store import insert_document, get_all_filenames, delete_document
from app.rag.chain import answer_query

router = APIRouter()

SUPPORTED_TYPES = {"image/png", "image/jpeg", "image/jpg", "application/pdf"}


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@router.post("/upload", summary="Upload a document (image or PDF)")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Supported: {SUPPORTED_TYPES}",
        )

    file_bytes = await file.read()
    filename = file.filename
    pages_processed = 0

    if file.content_type == "application/pdf":
        for page_num, img_bytes in pdf_to_images(file_bytes):
            content = extract_from_bytes(img_bytes, mime="image/png")
            if content:
                embedding = embed_text(content)
                insert_document(filename, page_num, content, embedding)
                pages_processed += 1
    else:
        content = extract_from_bytes(file_bytes, mime=file.content_type)
        if not content:
            raise HTTPException(status_code=422, detail="Could not extract content from image.")
        embedding = embed_text(content)
        insert_document(filename, page_num=1, content=content, embedding=embedding)
        pages_processed = 1

    return {
        "filename": filename,
        "pages_processed": pages_processed,
        "status": "indexed",
    }


@router.post("/query", summary="Ask a question across all uploaded documents")
def query_documents(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return answer_query(req.question, top_k=req.top_k)


@router.get("/documents", summary="List all indexed documents")
def list_documents():
    filenames = get_all_filenames()
    return {"documents": filenames, "count": len(filenames)}


@router.delete("/documents/{filename}", summary="Delete a document and its embeddings")
def remove_document(filename: str):
    delete_document(filename)
    return {"deleted": filename, "status": "ok"}
