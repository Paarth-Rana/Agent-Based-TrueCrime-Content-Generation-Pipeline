import re
import json
import requests
from urllib.parse import quote

BASE_OUT = "outputs"
MAX_SEARCH_QUERY_WORDS = 10
MIN_SEARCH_QUERY_WORDS = 3

FICTION_PATTERNS = re.compile(
    r"\b("
    r"television series|tv series|crime drama|drama series|miniseries|sitcom|"
    r"video game|computer game|novel|novella|fictional character|animated series|"
    r"premiered|aired on|created by|written by|directed by|"
    r"season \d+|episodes?|starring|"
    r"podcast series|film series|web series|"
    r"British crime drama|American crime drama|"
    r"video game franchise|role-playing game"
    r")\b",
    re.I,
)

INVALID_QUERY_PATTERNS = re.compile(
    r"("
    r"reported by the press|divorce from|daughter of|son of|husband |wife |"
    r"^\d{1,2},\s*\d{4}\.|was reported|according to sources|"
    r"in february|in january|in march|in april|in may|in june|"
    r"in july|in august|in september|in october|in november|in december"
    r")",
    re.I,
)

DESCRIPTION_FICTION_PATTERNS = re.compile(
    r"\b("
    r"series|drama|film|movie|novel|game|podcast|television|miniseries|sitcom"
    r")\b",
    re.I,
)

BLOCKED_CATEGORY_FRAGMENTS = (
    "television series",
    "television programmes",
    "television programs",
    "films",
    "video games",
    "novels",
    "fictional characters",
    "bbc programmes",
    "netflix",
    "hbo",
    "itv",
)


def clean_one_line(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[\r\n]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def normalize_query(q: str) -> str:
    q = clean_one_line(q)
    q = re.sub(r'^(search query:|query:)\s*', '', q, flags=re.I).strip()
    q = q.replace('"', '').replace("“", "").replace("”", "").strip()
    q = re.split(r"\b(SEARCH_QUERY|SEARCH QUERY|WIKI|WIKIPEDIA|LINK|URL)\b\s*:?", q, flags=re.I)[0].strip()
    return q


def sanitize_search_query(q: str, max_words: int = MAX_SEARCH_QUERY_WORDS) -> str:
    q = normalize_query(q)
    if not q:
        return ""
    words = q.split()
    if len(words) > max_words:
        q = " ".join(words[:max_words])
    return q.strip()


def is_plausible_search_query(q: str) -> bool:
    q = sanitize_search_query(q)
    if not q:
        return False
    words = q.split()
    if len(words) < MIN_SEARCH_QUERY_WORDS or len(words) > MAX_SEARCH_QUERY_WORDS:
        return False
    if INVALID_QUERY_PATTERNS.search(q):
        return False
    if is_fiction_text(q):
        return False
    return True


def is_fiction_text(text: str) -> bool:
    return bool(FICTION_PATTERNS.search(text or ""))


def is_fiction_description(description: str) -> bool:
    d = (description or "").strip()
    if not d:
        return False
    return bool(DESCRIPTION_FICTION_PATTERNS.search(d))


def is_valid_user_topic(q: str) -> bool:
    q = sanitize_search_query(q)
    if not q:
        return False
    if INVALID_QUERY_PATTERNS.search(q) or is_fiction_text(q):
        return False
    return True


def wiki_page_has_blocked_category(title: str) -> bool:
    title = normalize_title_or_url(title)
    if not title:
        return False
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "titles": title,
                "prop": "categories",
                "cllimit": 50,
                "format": "json",
            },
            headers={"User-Agent": "WikiLangGraphColab"},
            timeout=30,
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            for cat in page.get("categories", []):
                cat_title = (cat.get("title") or "").lower()
                if any(frag in cat_title for frag in BLOCKED_CATEGORY_FRAGMENTS):
                    return True
    except Exception:
        pass
    return False


def normalize_title_or_url(x: str) -> str:
    x = clean_one_line(x)
    m = re.search(r"wikipedia\.org\/wiki\/([^#\?]+)", x, flags=re.I)
    if m:
        return requests.utils.unquote(m.group(1)).replace("_", " ").strip()
    x = re.sub(r'^(title:)\s*', '', x, flags=re.I).strip()
    x = x.replace('"', '').replace("“", "").replace("”", "").strip()
    return x

def wiki_search(query: str, limit: int = 12):
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": limit},
        headers={"User-Agent": "WikiLangGraphColab"},
        timeout=30
    )
    r.raise_for_status()
    return r.json().get("query", {}).get("search", [])

def wiki_summary(title_or_url: str):
    title = normalize_title_or_url(title_or_url)
    if not title:
        return {"title": "", "extract": "", "url": "", "description": ""}

    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(title, safe="")
        r = requests.get(url, headers={"User-Agent": "WikiLangGraphColab"}, timeout=30)
        if r.status_code == 200:
            j = r.json()
            return {
                "title": j.get("title", title),
                "extract": (j.get("extract") or "").strip(),
                "url": (j.get("content_urls", {}).get("desktop", {}) or {}).get("page", "") or "",
                "description": (j.get("description") or "").strip(),
            }
    except Exception:
        pass

    return {"title": title, "extract": "", "url": "", "description": ""}

def extract_first_json_object(s: str):
    if not s:
        return None
    start = s.find("{")
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(s[start:])
        return obj
    except Exception:
        return None

def fallback_split(script: str, n: int):
    words = (script or "").split()
    if not words:
        return [{"section_text": ""} for _ in range(n)]
    per = max(70, len(words) // n)
    out = []
    for i in range(n):
        chunk = " ".join(words[i * per:(i + 1) * per]).strip()
        out.append({"section_text": chunk})
    return out
