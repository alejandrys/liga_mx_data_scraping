from playwright.sync_api import sync_playwright
import pandas as pd
import time
import random
from pathlib import Path

# =========================
# CONFIG
# =========================

BASE_URL = "https://www.oddsportal.com/football/mexico/liga-mx/results/"

OUTPUT_DIR = Path("data/raw/oddsportal")
HTML_DIR = OUTPUT_DIR / "html"

HTML_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# SCRAPER
# =========================

all_links = []

with sync_playwright() as p:

    browser = p.chromium.launch(
        headless=False
    )

    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    )

    page = context.new_page()

    print("Opening OddsPortal...")

    page.goto(BASE_URL)

    print("FINAL URL:", page.url)

    page.wait_for_timeout(5000)

    print("Page loaded.")

    # =========================
    # SAVE DEBUG HTML
    # =========================

    html = page.content()

    with open(
        "debug_page.html",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html)

    print("Debug HTML saved.")

    # =========================
    # EXTRACT MATCH LINKS
    # =========================

    links = page.locator("a").evaluate_all("""
        els => els
            .map(e => e.href)
            .filter(h =>
                h.includes('/football/mexico/liga-mx/')
            )
    """)

    links = list(set(links))

    print("\nFIRST 20 FILTERED LINKS:\n")

    for link in links[:20]:
        print(link)

    # =========================
    # SAVE URL CATALOG
    # =========================

    df_links = pd.DataFrame({
        "match_id": range(1, len(links) + 1),
        "url": links
    })

    urls_path = OUTPUT_DIR / "urls.csv"

    df_links.to_csv(
        urls_path,
        index=False
    )

    print(f"Saved URLs to: {urls_path}")

    # =========================
    # LIMIT INITIAL SCRAPING
    # =========================

    sample_links = links[:10]

    # =========================
    # DOWNLOAD MATCH HTML
    # =========================

    for idx, url in enumerate(sample_links):

        try:

            print(f"[{idx+1}] Visiting: {url}")

            page.goto(url)

            page.wait_for_timeout(
                random.randint(4000, 7000)
            )

            match_html = page.content()

            output_file = (
                HTML_DIR /
                f"match_{idx+1}.html"
            )

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(match_html)

            print(f"Saved: {output_file}")

            time.sleep(
                random.uniform(2, 5)
            )

        except Exception as e:

            print(f"Error scraping {url}")
            print(e)

    browser.close()

print("Phase 1 completed.")