from app.ingest.chunker import chunk_html

LONG = " ".join(["word"] * 40)

HTML = f"""
<html><body><div id="docContent">
  <h2>Chapter 5. Data Definition</h2>
  <h3>5.1. Table Basics</h3>
  <p>{LONG} tables are created with CREATE TABLE.</p>
  <ul><li><p>{LONG} nested paragraph inside a list item.</p></li></ul>
  <h3>5.2. Default Values</h3>
  <p>too short</p>
  <pre>CREATE TABLE products (
    price numeric DEFAULT 9.99
);</pre>
  <p>{LONG} defaults apply when no value is given.</p>
</div></body></html>
"""


def test_section_paths_and_chapter():
    chunks = chunk_html(HTML)
    assert chunks[0].chapter == "Chapter 5. Data Definition"
    assert chunks[0].section == "Chapter 5. Data Definition > 5.1. Table Basics"
    assert chunks[-1].section.endswith("5.2. Default Values")


def test_no_duplicate_text_from_nested_blocks():
    text = " ".join(c.text for c in chunk_html(HTML))
    assert text.count("nested paragraph inside a list item") == 1


def test_short_chunks_dropped_and_pre_kept():
    chunks = chunk_html(HTML)
    assert all(len(c.text.split()) >= 20 for c in chunks)
    assert "DEFAULT 9.99" in chunks[-1].text


def test_oversized_block_is_split():
    big = " ".join(f"w{i}" for i in range(500))
    html = f'<div id="docContent"><h2>T</h2><p>{big}</p></div>'
    chunks = chunk_html(html, max_words=200)
    assert len(chunks) == 3 and [c.index for c in chunks] == [0, 1, 2]
    assert all(len(c.text.split()) <= 200 for c in chunks)

def test_permalink_anchor_stripped_from_headings():
    html = (
        '<div id="docContent"><h2>5.1. Basics <a class="id_link" href="#x">#</a></h2>'
        f"<p>{LONG}</p></div>"
    )
    assert chunk_html(html)[0].section == "5.1. Basics"


def test_page_title_prefixes_reference_pages():
    html = (
        "<html><head><title>PostgreSQL: Documentation: 16: CREATE INDEX</title></head><body>"
        f'<div id="docContent"><h2>Notes</h2><p>{LONG}</p></div></body></html>'
    )
    c = chunk_html(html)[0]
    assert c.section == "CREATE INDEX > Notes" and c.chapter == "CREATE INDEX"