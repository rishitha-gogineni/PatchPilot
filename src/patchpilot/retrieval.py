"""Small, dependency-free repository retrieval for planner context."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
}
_SECRET_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
_SECRET_NAMES = {".env", "credentials", "secrets", "secret"}


@dataclass(frozen=True)
class RetrievedContext:
    path: str
    score: float
    excerpt: str


def _terms(value: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(value) if token.lower() not in _STOPWORDS]


def _is_safe_text_file(path: Path) -> bool:
    name = path.name.lower()
    return name not in _SECRET_NAMES and path.suffix.lower() not in _SECRET_SUFFIXES


def _excerpt(text: str, query_terms: set[str], max_chars: int) -> str:
    lines = text.splitlines()
    matching = [index for index, line in enumerate(lines) if query_terms.intersection(_terms(line))]
    if not matching:
        return text[:max_chars]
    selected: list[str] = []
    used = 0
    for index in matching[:8]:
        start = max(0, index - 2)
        end = min(len(lines), index + 3)
        block = "\n".join(lines[start:end]).strip()
        if not block or block in selected:
            continue
        if used + len(block) + 2 > max_chars:
            break
        selected.append(block)
        used += len(block) + 2
    return "\n...\n".join(selected)[:max_chars]


def retrieve_repository_context(
    repository: Path,
    query: str,
    allowed_files: Iterable[str],
    *,
    max_results: int = 8,
    max_chars_per_file: int = 2_500,
) -> list[RetrievedContext]:
    """Rank relevant text files with a bounded BM25-style lexical score."""
    if max_results < 1 or max_chars_per_file < 1:
        raise ValueError("retrieval limits must be positive")
    query_terms = set(_terms(query))
    if not query_terms:
        return []
    root = repository.expanduser().resolve()
    documents: list[tuple[str, str, list[str]]] = []
    for relative in allowed_files:
        path = (root / relative).resolve()
        if (root not in path.parents and path != root) or not path.is_file() or not _is_safe_text_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        terms = _terms(f"{relative} {text}")
        if terms:
            documents.append((relative, text, terms))
    if not documents:
        return []
    average_length = sum(len(terms) for _, _, terms in documents) / len(documents)
    document_frequency = {
        term: sum(1 for _, _, terms in documents if term in set(terms))
        for term in query_terms
    }
    scored: list[RetrievedContext] = []
    for relative, text, terms in documents:
        length = max(1, len(terms))
        frequencies = {term: terms.count(term) for term in query_terms}
        score = 0.0
        for term, frequency in frequencies.items():
            if not frequency:
                continue
            idf = math.log(1 + (len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * length / max(1.0, average_length))
            score += idf * (frequency * 2.5 / denominator)
        if score <= 0:
            continue
        scored.append(RetrievedContext(relative, round(score, 6), _excerpt(text, query_terms, max_chars_per_file)))
    scored.sort(key=lambda item: (-item.score, item.path))
    return scored[:max_results]
