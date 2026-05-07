from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from gepa_lab.data import Document

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\.]*")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from", "how",
    "if", "in", "is", "it", "of", "on", "or", "should", "the", "to", "use", "what",
    "when", "which", "who", "why", "with", "that", "this", "than", "only", "also",
}

SEMANTIC_EXPANSIONS = {
    "cost": {"expensive", "price", "operational", "cheaper", "budget"},
    "latency": {"p95", "slow", "seconds", "performance"},
    "search": {"retrieval", "rag", "intern", "enterprise"},
    "retrieval": {"search", "rag", "recall", "mrr"},
    "relationship": {"relations", "claims", "owned_by", "ownership", "entity", "entities"},
    "owner": {"owned", "owned_by", "team", "ownership"},
    "owns": {"owned", "owned_by", "owner", "ownership"},
    "support": {"supports", "capacity", "regions", "failover"},
    "approval": {"approve", "required", "requires", "mandatory"},
    "logs": {"records", "retained", "debug", "campaign"},
    "retained": {"retention", "preserved", "kept"},
    "metric": {"recall", "mrr", "precision", "groundedness", "correctness", "latency"},
    "metrics": {"recall", "mrr", "precision", "groundedness", "correctness", "latency"},
    "kg": {"epkg", "knowledge", "graph", "relations", "claims"},
    "epkg": {"kg", "knowledge", "graph", "relations", "claims"},
    "rl": {"reinforcement", "learning", "reliability", "label"},
    "qwen": {"model", "serving", "gpu", "p95"},
    "gpu": {"serving", "capacity", "model"},
}


@dataclass(frozen=True)
class SearchResult:
    doc: Document
    score: float
    tool_type: str
    rank: int


def tokenize(text: str, expand: bool = False) -> list[str]:
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    tokens = [t for t in tokens if t not in STOPWORDS]
    if expand:
        expanded = list(tokens)
        for token in tokens:
            expanded.extend(SEMANTIC_EXPANSIONS.get(token, set()))
        return expanded
    return tokens


class LexicalSearch:
    """Small BM25-ish lexical retriever."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.doc_tokens = [tokenize(d.text, expand=False) for d in docs]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / max(1, len(self.doc_len))
        self.df = defaultdict(int)
        for toks in self.doc_tokens:
            for tok in set(toks):
                self.df[tok] += 1

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if k <= 0:
            return []
        q_tokens = tokenize(query, expand=False)
        scores: list[tuple[float, int]] = []
        n = len(self.docs)
        for i, toks in enumerate(self.doc_tokens):
            tf = Counter(toks)
            score = 0.0
            for tok in q_tokens:
                if tok not in tf:
                    continue
                idf = math.log(1 + (n - self.df[tok] + 0.5) / (self.df[tok] + 0.5))
                numerator = tf[tok] * 2.2
                denominator = tf[tok] + 1.2 * (0.25 + 0.75 * self.doc_len[i] / max(1, self.avg_len))
                score += idf * numerator / denominator
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        return [SearchResult(self.docs[i], score, "lexical", rank + 1) for rank, (score, i) in enumerate(scores[:k])]


class SemanticSearch:
    """Tiny semantic-ish retriever using query expansion and overlap.

    This is intentionally not an embedding model so the lab runs without external
    dependencies or endpoints.
    """

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.doc_tokens = [set(tokenize(d.text, expand=True)) for d in docs]

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if k <= 0:
            return []
        q_tokens = set(tokenize(query, expand=True))
        scores: list[tuple[float, int]] = []
        for i, toks in enumerate(self.doc_tokens):
            if not q_tokens:
                score = 0.0
            else:
                overlap = len(q_tokens & toks)
                title_bonus = 0.25 * len(set(tokenize(self.docs[i].title, expand=True)) & q_tokens)
                entity_bonus = 0.35 * len(set(tokenize(" ".join(self.docs[i].entities), expand=True)) & q_tokens)
                score = overlap / math.sqrt(max(1, len(toks))) + title_bonus + entity_bonus
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        return [SearchResult(self.docs[i], score, "semantic", rank + 1) for rank, (score, i) in enumerate(scores[:k])]


class KnowledgeGraphTool:
    """Fictional EPKG-like relationship tool."""

    def __init__(self):
        self.claims = {
            "project falcon": [
                ("Project Falcon", "owned_by", "Infra Reliability"),
                ("Project Falcon", "supports", "GPU capacity planning"),
                ("Project Falcon", "deployed_in", "Ashburn"),
                ("Project Falcon", "deployed_in", "Prineville"),
            ],
            "epkg": [
                ("EPKG", "exposes", "entities"),
                ("EPKG", "exposes", "relations"),
                ("EPKG", "exposes", "claims"),
                ("EPKG", "exposes", "summaries"),
            ],
        }

    def search_claims(self, query: str) -> list[str]:
        q = query.lower()
        result: list[str] = []
        for entity, claims in self.claims.items():
            if entity in q or any(part in q for part in entity.split()):
                result.extend([f"{s} {p} {o}" for s, p, o in claims])
        if "relationship" in q or "claim" in q or "ownership" in q or "owned" in q:
            result.extend(["KG tools are preferred for entity relationships, ownership, and claims."])
        # preserve order and dedupe
        seen = set()
        deduped = []
        for item in result:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped


class SearchTools:
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.semantic = SemanticSearch(docs)
        self.lexical = LexicalSearch(docs)
        self.kg = KnowledgeGraphTool()

    def semantic_search(self, query: str, k: int) -> list[SearchResult]:
        return self.semantic.search(query, k=k)

    def lexical_search(self, query: str, k: int) -> list[SearchResult]:
        return self.lexical.search(query, k=k)

    def kg_claims(self, query: str) -> list[str]:
        return self.kg.search_claims(query)


def merge_results(result_sets: Iterable[list[SearchResult]], max_docs: int) -> list[SearchResult]:
    best_by_doc: dict[str, SearchResult] = {}
    for result_set in result_sets:
        for res in result_set:
            existing = best_by_doc.get(res.doc.doc_id)
            if existing is None or res.score > existing.score:
                best_by_doc[res.doc.doc_id] = res
    merged = list(best_by_doc.values())
    merged.sort(key=lambda r: (r.score, -r.rank), reverse=True)
    return merged[:max_docs]
