from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

# --- Load LLaMA Model --------------------------------------------------------
llm = Llama(model_path="models/llama-2-7b-chat.Q4_K_M.gguf", n_ctx=2048, verbose=False)

# --- Load Enriched Course Data ------------------------------------------------
with open("courses/all_courses_simple_enriched.json") as f:
    courses = json.load(f)
    texts = [f"{c['title']} — {c['enriched']['summary']}" for c in courses]

# --- Create Embeddings & Index -----------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(texts, convert_to_tensor=False)

dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# --- Some Keyword setup --------------------------------------------------
KEYWORDS = ["gender", "sexuality", "race", "identity", "power", "embodiment", "performance"]

# --- RAG Pipeline ------------------------------------------------------------
def recommend_courses(student_query, top_k=20, final_k=3):
    query_embedding = embedder.encode([student_query])[0]
    query_embedding_np = np.array([query_embedding])

    distances, indices = index.search(query_embedding_np, top_k)

    retrieved = [courses[i] for i in indices[0]]

    # FILTER HERE (before context)
    filtered = []
    for c in retrieved:
        text = (c["title"] + " " + c["enriched"]["summary"]).lower()
        if any(k in text for k in KEYWORDS):
            filtered.append(c)

    # If too few, fall back to best unfiltered (optional)
    final = (filtered[:final_k] if len(filtered) >= final_k else retrieved[:final_k])

    context = "\n".join([f"{c['title']}: {c['enriched']['summary']}" for c in final])

    print("##########")
    for r in retrieved:
        print("-", r['title'])
    prompt = f"""
        You are an academic advisor. A graduate student said: "{student_query}"

        Here are some available courses:

        {context}

        Which 2–3 courses should the student consider? Explain why they are relevant.
        Do not include ANY course that has a very short description such as Seminars and Independent Studies.
        """
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are an academic advisor."},
            {"role": "user", "content": f'A graduate student said: "{student_query}"\n\nHere are some available courses:\n\n{context}\n\nWhich 2–3 courses should the student consider? Explain why they are relevant.'}
        ],
        max_tokens=512,
    )
    reply = response["choices"][0]["message"]["content"].strip()

    return reply

student_input = input("Describe your research interest: ")
response = recommend_courses(student_input)
print("Recommended Courses:\n")
print(response)
