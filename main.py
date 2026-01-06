import json
import re
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama


# ----------------------------- Config ----------------------------------------
COURSES_PATH = "courses/all_courses_simple.json"
LLAMA_MODEL_PATH = "models/llama-2-7b-chat.Q4_K_M.gguf"

TERM = "Spring 2026"  # informational only; not used for filtering
TOP_K_RETRIEVE = 60   # retrieve more, then filter/rerank
FINAL_K = 3           # recommend up to this many

# Strict word-boundary patterns
GENDER_PATTERNS = [
    r"\bgender\b",
    r"\bmasculin\w*\b",
    r"\bfeminin\w*\b",
    r"\bwoman\b", r"\bwomen\b",
    r"\bman\b", r"\bmen\b",
    r"\bqueer\b",
    r"\blgbtq?\b", r"\blgbtqia\+?\b",
    r"\bsexualit\w*\b",
    r"\btrans\b", r"\btransgender\b",
    r"\bnonbinary\b", r"\bnon-binary\b",
]
THEATRE_PATTERNS = [
    r"\btheat(re|er)\b",
    r"\bperform\w*\b",
    r"\bacting\b",
    r"\bstage\b",
    r"\bdram(a|atic)\w*\b",
    r"\bplay\b", r"\bplays\b",
    r"\baudition\w*\b",
    r"\bscenograph\w*\b",
    r"\bchoreograph\w*\b",
    r"\brehears\w*\b",
]

gender_re = re.compile("|".join(GENDER_PATTERNS), re.IGNORECASE)
theatre_re = re.compile("|".join(THEATRE_PATTERNS), re.IGNORECASE)


def text_blob(course: Dict) -> str:
    return f"{course.get('title','')} — {course.get('description','')}"


def passes_strict_filter(course: Dict) -> bool:
    t = text_blob(course)
    return bool(gender_re.search(t)) and bool(theatre_re.search(t))


# ----------------------------- Load models ----------------------------------
llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=2048, verbose=False)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------- Load data ------------------------------------
with open(COURSES_PATH, "r", encoding="utf-8") as f:
    courses: List[Dict] = json.load(f)

# Embed title + description (critical)
corpus_texts = [text_blob(c) for c in courses]

# ------------------------- Build FAISS (cosine) ------------------------------
emb = embedder.encode(corpus_texts, convert_to_numpy=True).astype("float32")
faiss.normalize_L2(emb)

dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)  # cosine similarity after normalization
index.add(emb)


# ------------------------------ RAG -----------------------------------------
def retrieve(student_query: str, top_k: int = TOP_K_RETRIEVE) -> List[Dict]:
    q = embedder.encode([student_query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, top_k)
    return [courses[i] for i in idxs[0]]


def recommend_courses(student_query: str, final_k: int = FINAL_K) -> Tuple[List[Dict], str]:
    retrieved = retrieve(student_query, top_k=TOP_K_RETRIEVE)

    # show retrieval helps validate the index for debugging
    print("\n########## Retrieved candidates (top 15):")
    for r in retrieved[:15]:
        print("-", r["title"])

    # Strict filtering: must contain both gender + theatre/performance signals
    filtered = [c for c in retrieved if passes_strict_filter(c)]

    # If nothing passes, do not hallucinate relevance
    if not filtered:
        msg = (
            "No strong matches in this catalog.\n"
            "Reason: none of the retrieved course descriptions explicitly mention BOTH "
            "gender/sexuality-related terms AND theatre/performance-related terms."
        )
        return [], msg

    final = filtered[:final_k]

    # Build grounded context
    context = "\n".join([f"{c['title']}: {c['description']}" for c in final])

    user_msg = f"""
Student interest: "{student_query}"

ONLY recommend courses whose descriptions explicitly support the interest.
Pick up to {final_k} courses.

For EACH recommended course:
1) Quote ONE exact phrase from the description that proves relevance.
2) Give ONE short reason (1–2 sentences). No generic filler.

Courses:
{context}
""".strip()

    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are an academic advisor. Be strict, grounded, and concise."},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=512,
        temperature=0.2,
        top_p=0.9,
    )

    answer = resp["choices"][0]["message"]["content"].strip()
    return final, answer


if __name__ == "__main__":
    student_input = input("Describe your research interest: ").strip()
    recs, response = recommend_courses(student_input)

    print("\nRecommended Courses:\n")
    print(response)