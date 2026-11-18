"""Built-in internet tools: search (DuckDuckGo, no key) and page reading.

The LLM context is small: only short snippets are returned, and for a page only the passages
related to the question.

Tool descriptions and results are read by the model, so they stay in French.
"""

from __future__ import annotations

import json
import re
import unicodedata
from html.parser import HTMLParser

import httpx
from ddgs import DDGS

from .aio import in_thread

SEARCH_RESULTS = 5
SNIPPET_CHARS = 300
PAGE_CHARS = 2500
REGION = "fr-fr"
ACCEPT_LANGUAGE = "fr-FR,fr;q=0.9"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Cherche sur Internet. À utiliser pour toute information récente, locale ou que tu ne "
                "connais pas avec certitude : actualités, météo, bourse, horaires, séances de cinéma, "
                "restaurants, résultats sportifs, prix, événements… Renvoie des titres, des extraits et "
                "des adresses de pages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Mots-clés de la recherche, avec la ville si la demande est locale.",
                    },
                    "news": {
                        "type": "boolean",
                        "description": (
                            "true seulement pour les nouvelles du jour (« quelles sont les infos ? »). "
                            "false pour un cours de bourse, des horaires, des séances, la météo, un prix."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": (
                "Lit une page web trouvée par web_search quand ses extraits ne suffisent pas "
                "(cours exact, liste des séances, détails). Renvoie les passages de la page liés à la question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Adresse exacte reprise d'un résultat de web_search."},
                    "question": {"type": "string", "description": "Ce que tu cherches dans la page."},
                },
                "required": ["url", "question"],
            },
        },
    },
]


def _search(query: str, news: bool) -> str:
    ddgs = DDGS()
    if news:
        hits = ddgs.news(query, region=REGION, max_results=SEARCH_RESULTS)
    else:
        hits = ddgs.text(query, region=REGION, max_results=SEARCH_RESULTS)
    results = []
    for h in hits:
        item = {
            "title": h.get("title", ""),
            "snippet": (h.get("body") or "")[:SNIPPET_CHARS],
            "url": h.get("href") or h.get("url", ""),
        }
        if h.get("date"):
            item["date"] = h["date"]
        results.append(item)
    if not results:
        return "Aucun résultat."
    return json.dumps(results, ensure_ascii=False)


class _TextExtractor(HTMLParser):
    """Collects the visible text, split into blocks (paragraphs, table rows, headings…)."""

    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe", "template"}
    BLOCK = {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
             "td", "th", "dd", "dt", "br", "table", "ul", "ol", "span"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._buf: list[str] = []
        self._skip = 0
        self.title = ""
        self._in_title = False

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        if text:
            self.blocks.append(text)
        self._buf = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK and tag not in {"span", "td", "th"}:
            self._flush()
        elif tag in {"td", "th", "span"}:
            self._buf.append(" ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK and tag not in {"span", "td", "th"}:
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self._buf.append(data)

    def close(self):
        super().close()
        self._flush()


def _words(text: str) -> set[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return {w for w in re.findall(r"[a-z0-9]+", text) if len(w) > 2}


def _relevant(blocks: list[str], question: str, budget: int) -> str:
    """Keeps the blocks sharing the most words with the question, in page order."""
    blocks = list(dict.fromkeys(b for b in blocks if len(b) > 1))  # deduplicated
    if sum(len(b) for b in blocks) <= budget:
        return "\n".join(blocks)
    wanted = _words(question)
    scored = sorted(range(len(blocks)), key=lambda i: -len(wanted & _words(blocks[i])))
    keep: set[int] = set()
    used = 0
    for i in scored:
        # the next block often holds the figure or time that follows a heading
        for j in (i, i + 1):
            if j < len(blocks) and j not in keep and used + len(blocks[j]) <= budget:
                keep.add(j)
                used += len(blocks[j]) + 1
        if used >= budget:
            break
    return "\n".join(blocks[i] for i in sorted(keep))


async def _read_page(url: str, question: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "ERREUR : adresse invalide, reprends une url d'un résultat de web_search."
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=15, headers={"User-Agent": USER_AGENT, "Accept-Language": ACCEPT_LANGUAGE}
    ) as client:
        resp = await client.get(url)
    if resp.status_code >= 400:
        return f"ERREUR : la page a répondu {resp.status_code}. Essaie un autre résultat."
    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype:
        return f"ERREUR : ce n'est pas une page lisible ({ctype}). Essaie un autre résultat."
    parser = _TextExtractor()
    parser.feed(resp.text)
    parser.close()
    text = _relevant(parser.blocks, question, PAGE_CHARS)
    if not text:
        return "Page vide (sans doute affichée en JavaScript). Essaie un autre résultat."
    title = re.sub(r"\s+", " ", parser.title).strip()
    return f"{title}\n{text}" if title else text


async def call(name: str, args: dict) -> str:
    try:
        if name == "web_search":
            query = str(args.get("query") or "").strip()
            if not query:
                return "ERREUR : requête vide."
            return await in_thread(_search, query, bool(args.get("news")))
        if name == "read_page":
            return await _read_page(str(args.get("url", "")).strip(), str(args.get("question", "")))
    except Exception as e:  # network, DuckDuckGo rate limiting…
        return f"ERREUR : {e!r}"
    return f"ERREUR : outil inconnu {name}"
