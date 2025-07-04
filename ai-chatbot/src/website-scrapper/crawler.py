from dotenv import load_dotenv
import os
import json
from firecrawl import FirecrawlApp, ScrapeOptions
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import time
import requests

def main():
    # load secrets
    load_dotenv()
    firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not firecrawl_key or not openai_key:
        raise RuntimeError("firecrawl and openai api keys must be set")

    # init client
    app = FirecrawlApp(api_key=firecrawl_key)

    # start crawl (this returns a status response, not a bare id)
    status = app.crawl_url(
        "https://www.getsolar.ai/en-sg",
        limit=10,
        scrape_options=ScrapeOptions(formats=["markdown", "html"]),
    )
    print(f"initial status: {status.status} "
          f"({getattr(status, 'completed', '?')}/"
          f"{getattr(status, 'total', '?')})")

    # collect pages
    all_pages = [p.dict() for p in status.data]

    # if there's a `.next` url, fetch batches until exhausted
    next_url = getattr(status, "next", None)
    headers = {"Authorization": f"Bearer {firecrawl_key}"}

    while next_url:
        print(f"fetching next batch @ {next_url}")
        resp = requests.get(next_url, headers=headers)
        resp.raise_for_status()
        batch = resp.json()

        # extend pages
        all_pages.extend(batch.get("data", []))
        next_url = batch.get("next")

        # small backoff so you don’t hammer the API
        time.sleep(1)

    print(f"total pages fetched: {len(all_pages)}")

    # save raw crawl
    with open("../../data/scrape_result.json", "w") as f:
        json.dump({"data": all_pages}, f, indent=2)

    print("extraction complete, results in extracted_data.json")

if __name__ == "__main__":
    main()