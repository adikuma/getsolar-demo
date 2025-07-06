import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0, 
    google_api_key=os.getenv("GEMINI_API_KEY")
)

SYSTEM_PROMPT = """
you are a text-extraction tool. given html or markdown-html input, output
only the exact text content, preserving whitespace, blank lines, and all
characters. do not summarize, paraphrase, translate or alter any text.
strip all html tags, attributes, links, images, markdown formatting,
and emit only the raw text.
""".strip()

def extract_content(raw: str) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=raw)
    ]
    response = llm.invoke(messages)
    return response.content  # exact cleaned text


if __name__ == "__main__":
    in_path = os.path.join("..", "..", "data", "scrape_result.json")
    out_path = os.path.join("..", "..", "data", "cleaned_scrape_result.json")

    with open(in_path, "r", encoding="utf8") as f:
        raw = json.load(f)

    for i, page in enumerate(raw["data"], start=1):
        html_blob = page.get("markdown", "")
        if not html_blob:
            continue  

        cleaned = extract_content(html_blob)
        page["markdown"] = cleaned

        print(f"[{i}/{len(raw['data'])}] cleaned page, {len(cleaned)} chars")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    print(f"wrote cleaned data to {out_path}")