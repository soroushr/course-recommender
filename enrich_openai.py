import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


INPUT_JSON = "courses/all_courses_simple.json"
OUTPUT_JSON = "courses/all_courses_simple_enriched.json"

client = OpenAI()

PROMPT = """You are enriching a university course catalog.

Return VALID JSON only with exactly these keys:
- summary: 2–3 sentences, faithful to the description (do not add facts)
- themes: 3–6 short noun-phrase labels
- domains: academic fields
- methods: approaches implied by the description

Course title: {title}
Course description: {description}
"""

def enrich_one(title: str, description: str) -> dict:
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=PROMPT.format(title=title, description=description),
        # Ask for JSON output (so parsing is reliable)
        text={"format": {"type": "json_object"}},
    )
    # Responses API returns output text content; parse it as JSON
    content = resp.output_text
    return json.loads(content)

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        courses = json.load(f)

    out = []
    for i, c in enumerate(courses, 1):
        print(f"{i}/{len(courses)} Enriching: {c['title']}")
        enriched = enrich_one(c["title"], c["description"])
        out.append({**c, "enriched": enriched})

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f" Wrote {OUTPUT_JSON}")

if __name__ == "__main__":
    main()