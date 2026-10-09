import re
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

BASE = "https://www.postgresql.org/docs/{version}/"
PAGE_RE = re.compile(r"^[a-z0-9\-]+\.html$")
SKIP_PREFIXES = ("release", "bookindex", "legalnotice", "docguide", "sourcerepo", "acronyms")
HEADERS = {"User-Agent": "pgdocs-assistant/0.1 (portfolio project)"}


def _make_client() -> httpx.Client:
    # No keep-alive: a fresh connection per request avoids "server disconnected" errors
    return httpx.Client(
        timeout=30,
        headers=HEADERS,
        limits=httpx.Limits(max_keepalive_connections=0),
    )


def _fetch(client: httpx.Client, url: str, retries: int = 4) -> tuple[httpx.Client, str | None]:
    """GET with retries and exponential backoff. Returns (client, html or None)."""
    for attempt in range(retries):
        try:
            resp = client.get(url)
            if resp.status_code == 200:
                return client, resp.text
            if resp.status_code == 404:
                return client, None
            # 429 / 5xx: wait and retry
        except httpx.HTTPError as e:
            print(f"  retry {attempt + 1}/{retries} for {url}: {type(e).__name__}", flush=True)
            client.close()
            client = _make_client()  # start with a clean connection
        time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s
    return client, None


def crawl(version: str, cache_dir: Path, max_pages: int | None = None, delay: float = 0.6):
    """Yield (url, html) for each docs page, breadth-first from index.html."""
    base = BASE.format(version=version)
    cache = cache_dir / version
    cache.mkdir(parents=True, exist_ok=True)
    queue, seen = ["index.html"], {"index.html"}
    n = 0
    client = _make_client()
    try:
        while queue and (max_pages is None or n < max_pages):
            name = queue.pop(0)
            f = cache / name
            if f.exists():                          # cached: no network needed
                html = f.read_text(encoding="utf-8")
            else:
                client, html = _fetch(client, urljoin(base, name))
                if html is None:
                    print(f"  skipped {name}", flush=True)
                    continue
                f.write_text(html, encoding="utf-8")
                time.sleep(delay)                   # be polite to the server
            n += 1
            yield urljoin(base, name), html
            for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
                href = a["href"].split("#")[0]
                if PAGE_RE.match(href) and href not in seen and not href.startswith(SKIP_PREFIXES):
                    seen.add(href)
                    queue.append(href)
    finally:
        client.close()