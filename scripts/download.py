"""Download the GDPR and EU AI Act (official English HTML) from EUR-Lex into data/raw/.

Usage:  python scripts/download.py
If EUR-Lex blocks the request, open the URLs in a browser, "Save page as -> HTML only",
and save them as data/raw/gdpr.html and data/raw/ai_act.html.
"""

from pathlib import Path

import httpx

SOURCES = {
    "gdpr": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679",
    "ai_act": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (eu-reg-rag research project)"}


def main() -> None:
    raw = Path("data/raw")
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        out = raw / f"{name}.html"
        print(f"Downloading {name} ...")
        r = httpx.get(url, headers=HEADERS, timeout=60, follow_redirects=True)
        r.raise_for_status()
        if "Article 1" not in r.text:
            raise SystemExit(
                f"{name}: response doesn't look like the regulation text "
                f"(maybe a bot check). Save it manually from {url}"
            )
        out.write_text(r.text, encoding="utf-8")
        print(f"  saved {out} ({len(r.text) // 1024} KB)")


if __name__ == "__main__":
    main()
