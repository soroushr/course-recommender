# course-recommender

This repo is a hands-on exploration of building a lightweight **LLM-powered course recommendation system** using RAG. The goal is not to productionize a full application, but to experiment with realistic AI workflows and demonstrate practical understanding of embeddings, vector search, and LLM prompting.

The project uses real university course catalog data and progressively improves recommendation quality through **semantic search and description enrichment**.

---

## Repository Structure

### `courses/`
This directory contains all course catalog data in JSON format.

- Raw course data is stored in a simple, uniform schema (`title`, `description`).
- Enriched versions of the same data include expanded, more semantically meaningful descriptions generated using an LLM.
- Keeping the data separate from code makes it easy to swap catalogs, experiment with enrichment strategies, and re-index without changing logic.

---

### `models/`
Local model artifacts used for experimentation (e.g., local LLaMA weights or related files). This allows the project to run fully offline when desired.

---

### `main.py`
This is the **core RAG pipeline**:
- Loads course data
- Creates embeddings
- Indexes them using a vector database (FAISS)
- Accepts a user research-interest query
- Retrieves semantically similar courses
- Uses an LLM to generate a recommendation explanation

This file demonstrates the end-to-end retrieval + generation workflow.

---

### `enrich_openai.py`
This script exists to **enrich course descriptions before retrieval**.

Short catalog descriptions are often too sparse for high-quality semantic search. This script uses an LLM (via the OpenAI API) to expand each course description with clearer academic framing and richer semantic signal.

The enriched output is saved as a new JSON file and used directly by the RAG pipeline. This separation makes it easy to compare retrieval quality *before vs. after enrichment*.

---

## Why Enrichment Matters

This project intentionally highlights a common real-world issue: **LLMs and vector search perform poorly on sparse or generic text**.

By enriching descriptions offline:
- embeddings become more informative
- retrieval improves without changing the model
- the system scales better across departments without manual keyword rules

This mirrors how real production RAG systems often preprocess and augment data before indexing.

---

## Scope and Intent

This project is designed for:
- learning by building
- experimentation with local vs API-based models
- portfolio demonstration for AI / Data Science roles

It is intentionally minimal on UI and infrastructure, and heavy on **conceptual correctness and clarity**.
