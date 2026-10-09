import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

HEADINGS = ("h1", "h2", "h3", "h4")
BLOCKS = ["p", "pre", "li", "dt", "dd", "td"]
MAX_WORDS = 220
MIN_WORDS = 20

# <title> looks like "PostgreSQL: Documentation: 16: CREATE INDEX"
TITLE_PREFIX = re.compile(r"^PostgreSQL:\s*Documentation:\s*\S+:\s*")
TRAILING_HASH = re.compile(r"\s*#\s*$")


@dataclass
class Chunk:
    chapter: str
    section: str
    text: str
    index: int = 0

    @property
    def embed_text(self) -> str:
        # Prepending the section path gives the embedding extra context
        return f"{self.section}\n{self.text}"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _heading_text(el) -> str:
    """Heading text without the trailing '#' permalink marker."""
    return TRAILING_HASH.sub("", _clean(el.get_text(" ")))


def _pack(blocks: list[str], max_words: int) -> list[str]:
    """Group blocks into windows of at most max_words; split oversized blocks."""
    out: list[str] = []
    cur: list[str] = []
    count = 0
    for block in blocks:
        words = block.split()
        if len(words) > max_words:
            if cur:
                out.append("\n".join(cur))
                cur, count = [], 0
            for i in range(0, len(words), max_words):
                out.append(" ".join(words[i : i + max_words]))
            continue
        if count + len(words) > max_words and cur:
            out.append("\n".join(cur))
            cur, count = [], 0
        cur.append(block)
        count += len(words)
    if cur:
        out.append("\n".join(cur))
    return out


def chunk_html(html: str, max_words: int = MAX_WORDS, min_words: int = MIN_WORDS) -> list[Chunk]:
    soup = BeautifulSoup(html, "html.parser")

    # Reference pages (e.g. CREATE INDEX) keep their name in <title>, not in a heading
    page_title = ""
    if soup.title and soup.title.string:
        page_title = TITLE_PREFIX.sub("", _clean(soup.title.string))

    root = soup.find(id="docContent") or soup.body or soup
    for tag in root.find_all(["script", "style", "nav"]):
        tag.decompose()
    for a in root.find_all("a", class_="id_link"):  # "#" permalink anchors
        a.decompose()

    stack: list[tuple[int, str]] = []           # heading hierarchy: (level, title)
    chapter = page_title
    sections: list[tuple[str, list[str]]] = []  # (section_path, text blocks)
    current: list[str] = []
    path = ""

    def flush() -> None:
        if current:
            sections.append((path, list(current)))
            current.clear()

    for el in root.find_all(HEADINGS + tuple(BLOCKS)):
        if el.name in HEADINGS:
            title = _heading_text(el)
            if not title:
                continue
            flush()
            level = int(el.name[1])
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            if not chapter:
                chapter = title
            titles = [t for _, t in stack]
            if page_title and titles[0] != page_title:
                titles.insert(0, page_title)    # "CREATE INDEX > Notes"
            path = " > ".join(titles)
            continue
        if el.find(BLOCKS):  # skip outer blocks that contain inner ones (li > p)
            continue
        text = el.get_text() if el.name == "pre" else _clean(el.get_text(" "))
        text = text.strip()
        if text:
            current.append(text)
    flush()

    chunks: list[Chunk] = []
    for sec_path, blocks in sections:
        for text in _pack(blocks, max_words):
            if len(text.split()) >= min_words:
                chunks.append(Chunk(chapter=chapter, section=sec_path or chapter, text=text))
    for i, c in enumerate(chunks):
        c.index = i
    return chunks