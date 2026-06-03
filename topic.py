import re
import random
from typing import Dict, Any

from utils import (
    clean_one_line,
    normalize_title_or_url,
    sanitize_search_query,
    wiki_search,
    wiki_summary,
)

BLOCKED_NS_PREFIXES = (
    "User:", "Wikipedia:", "Template:", "Draft:", "Talk:", "File:",
    "Help:", "Category:", "Portal:", "Module:", "TimedText:", "Media:",
    "Special:", "Book:", "Education Program:", "Gadget:", "Thread:",
)

FALLBACK_TITLE = "D. B. Cooper"
MIN_SUMMARY_CHARS = 80
MAX_RANDOM_TRIES = 8

DISAMBIGUATION = re.compile(r"disambiguation", re.I)

CURATED_CASES = (
    "D. B. Cooper",
    "Ted Bundy",
    "Watergate scandal",
    "Enron scandal",
    "Zodiac Killer",
    "Murder of Elizabeth Short",
    "Lindbergh kidnapping",
    "Bonnie and Clyde",
    "O. J. Simpson murder case",
    "Elizabeth Holmes",
    "Ted Kaczynski",
    "John Wayne Gacy",
    "Boston Marathon bombing",
    "Dennis Rader",
    "Jack the Ripper",
    "Lizzie Borden",
    "Al Capone",
    "Bernie Madoff",
    "Golden State Killer",
    "Leopold and Loeb",
    "Sam Bankman-Fried",
    "Murder of JonBenét Ramsey",
    "Waco siege",
    "Pan Am Flight 103",
    "Disappearance of Malaysia Airlines Flight 370",
    "Assassination of John F. Kennedy",
    "Kidnapping of Patty Hearst",
    "Great Train Robbery (1963)",
    "Isabella Stewart Gardner Museum theft",
    "Murder of Tupac Shakur",
    "Murder of Gianni Versace",
    "Harvey Weinstein sexual abuse cases",
    "Volkswagen emissions scandal",
    "Charles Manson",
    "Jeffrey Dahmer",
)

_CURATED_BY_LOWER = {t.lower(): t for t in CURATED_CASES}


def is_blocked_namespace(title: str) -> bool:
    t = (title or "").strip()
    if not t:
        return True
    for prefix in BLOCKED_NS_PREFIXES:
        if t.startswith(prefix):
            return True
    if "/drafts/" in t.lower():
        return True
    return False


def is_usable_wikipedia_article(info: Dict[str, Any]) -> bool:
    title = clean_one_line(info.get("title") or "")
    extract = (info.get("extract") or "").strip()
    description = clean_one_line(info.get("description") or "")
    if not title or is_blocked_namespace(title):
        return False
    if len(extract) < MIN_SUMMARY_CHARS:
        return False
    if DISAMBIGUATION.search(title) or DISAMBIGUATION.search(description):
        return False
    return True


def _article_state(user_query: str, search_query: str, info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query": user_query or search_query,
        "search_query": search_query,
        "source_title": info["title"],
        "source_url": info["url"],
        "source_text": info["extract"],
    }


def _resolve_title(title: str, user_query: str, search_query: str) -> Dict[str, Any]:
    info = wiki_summary(title)
    if is_usable_wikipedia_article(info):
        return _article_state(user_query, search_query, info)
    return {}


def _pick_from_user_query(user_query: str) -> Dict[str, Any]:
    q = sanitize_search_query(user_query)
    if not q:
        return {}

    q_lower = q.lower()
    for title in CURATED_CASES:
        t_lower = title.lower()
        if q_lower == t_lower or q_lower in t_lower or t_lower in q_lower:
            resolved = _resolve_title(title, user_query=q, search_query=q)
            if resolved:
                return resolved

    resolved = _resolve_title(q, user_query=q, search_query=q)
    if resolved:
        return resolved

    for hit in wiki_search(q, limit=10):
        title = normalize_title_or_url((hit.get("title") or "").strip())
        canonical = _CURATED_BY_LOWER.get(title.lower())
        if canonical:
            resolved = _resolve_title(canonical, user_query=q, search_query=q)
            if resolved:
                return resolved

    return {}


def _pick_random_curated(user_query: str) -> Dict[str, Any]:
    candidates = list(CURATED_CASES)
    random.shuffle(candidates)
    for title in candidates[:MAX_RANDOM_TRIES]:
        resolved = _resolve_title(title, user_query=user_query, search_query=title)
        if resolved:
            return resolved
    return {}


def node_discover_topic(state: Dict[str, Any]) -> Dict[str, Any]:
    user_query = clean_one_line(state.get("query") or "")

    if user_query:
        resolved = _pick_from_user_query(user_query)
        if resolved:
            return resolved

    resolved = _pick_random_curated(user_query)
    if resolved:
        return resolved

    info = wiki_summary(FALLBACK_TITLE)
    return _article_state(user_query, FALLBACK_TITLE, info)


def discover_topic(query: str = "") -> Dict[str, Any]:
    return node_discover_topic({"query": query})


def validate_selection(state: Dict[str, Any]) -> Dict[str, Any]:
    title = clean_one_line(state.get("source_title") or "")
    extract = (state.get("source_text") or "").strip()
    search_query = clean_one_line(state.get("search_query") or "")

    in_curated = title.lower() in _CURATED_BY_LOWER or search_query.lower() in _CURATED_BY_LOWER
    has_summary = len(extract) >= MIN_SUMMARY_CHARS

    return {
        "in_curated": in_curated,
        "has_summary": has_summary,
        "ok": in_curated and has_summary,
    }
