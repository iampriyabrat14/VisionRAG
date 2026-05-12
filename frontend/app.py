import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="VisionRAG", page_icon="🔍", layout="wide")
st.title("VisionRAG — Multimodal Document Intelligence")
st.caption("Upload scanned PDFs, invoices, images, or screenshots and ask questions.")

tab_upload, tab_query, tab_docs = st.tabs(["Upload", "Ask Questions", "Manage Documents"])

# ── Upload Tab ──────────────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload a Document")
    uploaded = st.file_uploader(
        "Supported: PDF, PNG, JPG",
        type=["pdf", "png", "jpg", "jpeg"],
    )

    if uploaded and st.button("Index Document"):
        with st.spinner(f"Extracting and indexing {uploaded.name}..."):
            resp = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            )
        if resp.status_code == 200:
            data = resp.json()
            st.success(
                f"Indexed **{data['filename']}** — {data['pages_processed']} page(s) processed."
            )
        else:
            st.error(f"Upload failed: {resp.text}")

# ── Query Tab ────────────────────────────────────────────────────────────────
with tab_query:
    st.subheader("Ask a Question")
    question = st.text_area("Your question", placeholder="What is the total amount on the invoice?")
    top_k = st.slider("Number of context chunks", min_value=1, max_value=10, value=5)

    if st.button("Get Answer") and question.strip():
        with st.spinner("Searching and generating answer..."):
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": question, "top_k": top_k},
            )
        if resp.status_code == 200:
            data = resp.json()

            st.markdown("### Answer")
            st.write(data["answer"])

            st.markdown("### Sources")
            for c in data["citations"]:
                st.markdown(
                    f"- **{c['source']}** — Page {c['page']} "
                    f"*(relevance: {c['score']:.2%})*"
                )

            with st.expander("Show retrieved context chunks"):
                for i, chunk in enumerate(data["context_chunks"], start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk)
        else:
            st.error(f"Query failed: {resp.text}")

# ── Documents Tab ────────────────────────────────────────────────────────────
with tab_docs:
    st.subheader("Indexed Documents")

    if st.button("Refresh"):
        st.rerun()

    resp = requests.get(f"{API_URL}/documents")
    if resp.status_code == 200:
        docs = resp.json()["documents"]
        if docs:
            for doc in docs:
                col1, col2 = st.columns([4, 1])
                col1.write(doc)
                if col2.button("Delete", key=doc):
                    del_resp = requests.delete(f"{API_URL}/documents/{doc}")
                    if del_resp.status_code == 200:
                        st.success(f"Deleted {doc}")
                        st.rerun()
        else:
            st.info("No documents indexed yet. Upload one to get started.")
    else:
        st.error("Could not reach API.")
