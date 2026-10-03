"""Static checks of the scouting site's pages, stylesheets and scripts."""

import re
import struct
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
SITE = "https://abhigyanrathi.github.io/football-scout/"
PAGES = {"index.html": SITE, "about.html": SITE + "about.html"}
LOGO = "assets/hudl-statsbomb-logo-reversed.png"
PREVIEW = "assets/preview.png"
# what a page's head says of its preview image, by the property or name of each meta tag
CARD = {
    "og:image": SITE + PREVIEW,
    "og:image:width": "1200",
    "og:image:height": "630",
    "og:image:alt": (
        "Football Scout's page for Leicester City: "
        "the club's style fingerprint beside its eight style scores."
    ),
    "twitter:card": "summary_large_image",
}


class Tags(HTMLParser):
    """Every start tag of a page with its attributes, and the text of its title."""

    def __init__(self):
        super().__init__()
        self.tags = []
        self.title = ""

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        if self.lasttag == "title" and not self.title:
            self.title = data.strip()


def parse(page):
    parser = Tags()
    parser.feed((WEB / page).read_text(encoding="utf-8"))
    return parser


def files(*suffixes):
    return sorted(p for p in WEB.rglob("*") if p.suffix in suffixes)


def local(source, target):
    """The file under web/ that a relative address in source leads to, or None if it leaves the
    site or leads nowhere. A folder stands for its index.html, and a bare fragment for source."""
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or target.startswith("//"):
        return None
    path = (source.parent / parts.path).resolve() if parts.path else source
    if path.is_dir():
        path = path / "index.html"
    return path if path.is_file() and WEB in path.parents else None


def test_the_site_is_its_two_pages():
    assert {p.relative_to(WEB).as_posix() for p in files(".html")} == set(PAGES)


@pytest.mark.parametrize("page, address", PAGES.items())
def test_a_page_has_its_head_and_the_logo(page, address):
    parsed = parse(page)
    found = {}
    for tag, attrs in parsed.tags:
        found.setdefault(tag, []).append(attrs)
    assert found["html"][0].get("lang") == "en"
    assert parsed.title
    assert [m["content"] for m in found["meta"] if m.get("name") == "viewport"]
    assert [x["href"] for x in found["link"] if x.get("rel") == "canonical"] == [address]
    assert [m["content"] for m in found["meta"] if m.get("property") == "og:url"] == [address]
    for key, content in CARD.items():
        said = [m["content"] for m in found["meta"] if key in (m.get("property"), m.get("name"))]
        assert said == [content], key
    # the preview image is there, at the size the head gives for it, from its header
    assert (WEB / PREVIEW).is_file()
    assert struct.unpack(">II", (WEB / PREVIEW).read_bytes()[16:24]) == (1200, 630)
    logos = [i for i in found["img"] if i.get("src") == LOGO]
    assert len(logos) == 1 and logos[0].get("alt") == "Hudl StatsBomb"
    # the size attributes are the file's own pixels, from its header
    size = struct.unpack(">II", (WEB / LOGO).read_bytes()[16:24])
    assert (int(logos[0]["width"]), int(logos[0]["height"])) == size


@pytest.mark.parametrize("page, address", PAGES.items())
def test_a_page_loads_nothing_from_another_origin(page, address):
    source = WEB / page
    for tag, attrs in parse(page).tags:
        assert "srcset" not in attrs
        for name in ("src", "href"):
            if name not in attrs:
                continue
            target = attrs[name]
            # only a link for the reader to follow, or the canonical address, may leave the site
            outbound = name == "href" and (tag == "a" or attrs.get("rel") == "canonical")
            if outbound and urlsplit(target).scheme == "https":
                continue
            assert local(source, target), f"{page}: {tag} {name}={target}"
    # and no other address appears anywhere in the page, in a script or a style, but that of the
    # preview image, which the head must give in full
    text = source.read_text(encoding="utf-8")
    allowed = {a["href"] for t, a in parse(page).tags if t == "a" and "href" in a}
    allowed |= {address, SITE + PREVIEW}
    for found in re.findall(r"""(?:[a-z]+:)?//[^\s"'<>)]+""", text):
        assert found in allowed, f"{page}: {found}"


def test_no_stylesheet_reaches_another_origin():
    sheets = files(".css")
    assert sheets
    for sheet in sheets:
        text = sheet.read_text(encoding="utf-8")
        targets = re.findall(r"""url\(\s*["']?([^"')]+)""", text)
        targets += re.findall(r"""@import\s+(?:url\(\s*)?["']?([^"');\s]+)""", text)
        for target in targets:
            assert local(sheet, target), f"{sheet.name}: {target}"


def test_no_script_names_another_origin():
    scripts = files(".js")
    assert scripts
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "http://" not in text and "https://" not in text, script.name
