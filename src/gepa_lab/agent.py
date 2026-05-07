from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from time import perf_counter

from gepa_lab.config import ModelProfile, PromptPolicy
from gepa_lab.data import Document
from gepa_lab.retrieval import SearchResult, SearchTools, merge_results, tokenize


@dataclass(frozen=True)
class ToolCall:
    tool_type: str
    query: str
    k: int | None = None
    result_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AgentRun:
    question: str
    answer: str
    citations: tuple[str, ...]
    retrieved_doc_ids: tuple[str, ...]
    tool_calls: tuple[ToolCall, ...]
    kg_claims: tuple[str, ...]
    policy: PromptPolicy
    model: ModelProfile
    latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": list(self.citations),
            "retrieved_doc_ids": list(self.retrieved_doc_ids),
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "kg_claims": list(self.kg_claims),
            "policy": self.policy.to_dict(),
            "model": asdict(self.model),
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


def _looks_exact_or_identifier(question: str) -> bool:
    return bool(re.search(r"\b[A-Z]{2,}\b|\$|\d|Qwen|Falcon|EPKG|ACL|SEV2|p95|RL", question))


def _looks_relationship_question(question: str) -> bool:
    q = question.lower()
    terms = [
        "relationship", "relationships", "claim", "claims", "owned", "owner", "owns", "support",
        "supports", "team", "regions", "kg", "epkg", "falcon",
    ]
    return any(term in q for term in terms)


def _rewrite_query(question: str) -> str:
    q = question
    lower = q.lower()
    additions: list[str] = []
    if "budget" in lower or "$250" in lower or "250k" in lower:
        additions.extend(["FinanceOps", "CFO approval", "budget transfer", "threshold"])
    if "falcon" in lower:
        additions.extend(["Project Falcon", "Infra Reliability", "GPU capacity", "Ashburn", "Prineville", "owned_by", "supports"])
    if "relationship" in lower or "claim" in lower or "ownership" in lower:
        additions.extend(["EPKG", "knowledge graph", "entities", "relations", "claims", "summaries"])
    if "metric" in lower or "evaluate" in lower:
        additions.extend(["Recall@5", "Recall@10", "MRR", "precision", "groundedness", "latency", "tool calls"])
    if "p95" in lower or "latency" in lower or "incident" in lower:
        additions.extend(["SEV2", "Search Reliability", "8 seconds", "on-call", "error rate"])
    if "wellness" in lower or "yoga" in lower:
        additions.extend(["wellness reimbursement", "$750", "policy does not mention yoga mats"])
    if "rl" in lower:
        additions.extend(["Reinforcement Learning", "Reliability Label", "acronym disambiguation"])
    if "qwen" in lower or "production" in lower or "model" in lower:
        additions.extend(["Qwen 3.5 110B", "custom GPU serving", "capacity planning", "p95 latency"])
    if "acl" in lower or "permission" in lower or "snippet" in lower:
        additions.extend(["document-level ACL", "permissions", "before LLM sees snippets"])
    if not additions:
        return question
    return q + " | expanded terms: " + "; ".join(dict.fromkeys(additions))


def _truncate_docs(docs: list[SearchResult], max_docs: int) -> list[Document]:
    return [res.doc for res in docs[:max_docs]]


def _has_doc(documents: list[Document], doc_id: str) -> bool:
    return any(d.doc_id == doc_id for d in documents)


def _context_text(documents: list[Document], kg_claims: list[str]) -> str:
    doc_text = "\n".join(f"[{d.doc_id}] {d.text}" for d in documents)
    kg_text = "\n".join(f"[KG] {claim}" for claim in kg_claims)
    return doc_text + ("\n" + kg_text if kg_text else "")


def _cite(answer: str, citations: list[str], policy: PromptPolicy) -> tuple[str, tuple[str, ...]]:
    if not policy.require_citations or not citations:
        return answer, tuple()
    unique = list(dict.fromkeys(citations))
    return answer.rstrip() + " " + " ".join(f"[{c}]" for c in unique), tuple(unique)


def _local_generate_answer(question: str, documents: list[Document], kg_claims: list[str], policy: PromptPolicy) -> tuple[str, tuple[str, ...]]:
    """Rule-based generator so the lab is runnable without model APIs.

    It intentionally behaves better when the optimized policy retrieves the right
    docs, requires citations, uses KG for relationship questions, and abstains
    on unsupported answers.
    """

    q = question.lower()
    ctx = _context_text(documents, kg_claims).lower()
    citations: list[str] = []
    answer = "I could not find enough supported context to answer."

    def cite_doc(doc_id: str) -> None:
        if _has_doc(documents, doc_id):
            citations.append(doc_id)

    if "campaign logs" in q and "90 days" in ctx:
        answer = "Campaign logs are retained for 90 days."
        cite_doc("DOC-001")
    elif "training-debug" in q and "30 days" in ctx:
        answer = "Training-debug logs are retained for 30 days."
        cite_doc("DOC-001")
    elif ("budget" in q or "250k" in q or "$250" in q) and "cfo approval" in ctx:
        if "over" in q or "above" in q or "$250" in q or "250k" in q:
            answer = "Budget transfers over $250k require CFO approval."
        else:
            answer = "Transfers under $250k need manager and finance approval; transfers over $250k require CFO approval."
        cite_doc("DOC-002")
    elif "falcon" in q and ("infra reliability" in ctx or any("Infra Reliability" in c for c in kg_claims)):
        if policy.use_kg_for_relationships and kg_claims:
            if "regions" in q or "ashburn" in ctx or "prineville" in ctx:
                answer = "Project Falcon is owned by Infra Reliability and supports GPU capacity planning in Ashburn and Prineville."
            else:
                answer = "Project Falcon is owned by Infra Reliability and supports GPU capacity planning and failover."
            cite_doc("DOC-003")
            cite_doc("DOC-014")
        else:
            answer = "Project Falcon is owned by Infra Reliability."
            cite_doc("DOC-003")
    elif ("relationship claims" in q or "metrics" in q or "evaluate" in q) and "recall" in ctx:
        if policy.answer_style == "synthesis" or policy.use_kg_for_relationships:
            answer = (
                "Use EPKG/KG tools for relationship claims. Evaluate the search agent with Recall@5/10, "
                "MRR, precision, correctness, completeness, groundedness, latency, and average tool calls."
            )
            cite_doc("DOC-004")
            cite_doc("DOC-005")
            cite_doc("DOC-006")
        else:
            answer = "Evaluate with recall, MRR, and groundedness."
            cite_doc("DOC-005")
    elif "exact policy codes" in q or "ownership claims" in q:
        if "lexical" in ctx and ("kg" in ctx or "epkg" in ctx):
            answer = "Use lexical search for exact policy codes and KG/EPKG tools for ownership claims or relationship reasoning."
            cite_doc("DOC-006")
            cite_doc("DOC-004")
        else:
            answer = "Use semantic search for general retrieval."
    elif "p95" in q and "8 seconds" in ctx and "sev2" in ctx:
        answer = (
            "If search p95 latency exceeds 8 seconds for 10 consecutive minutes, file a SEV2 in the Search Reliability queue; "
            "page on-call only if user-facing traffic is affected or error rate exceeds 2%."
        )
        cite_doc("DOC-009")
    elif "search on-call" in q and "error rate exceeds 2%" in ctx:
        answer = "Page the search on-call only if user-facing traffic is affected or error rate exceeds 2%."
        cite_doc("DOC-009")
    elif "wellness" in q and "yoga mats" in q:
        cite_doc("DOC-011")
        if "does not mention yoga mats" in ctx and policy.abstain_when_low_confidence:
            answer = "No. The wellness policy does not explicitly mention yoga mats."
        elif "wellness reimbursement" in ctx:
            answer = "Yoga mats appear to be eligible under the wellness reimbursement policy."
        else:
            answer = "I could not find enough supported context to answer."
    elif "wellness reimbursement maximum" in q and "$750" in ctx:
        answer = "The annual wellness reimbursement maximum is $750."
        cite_doc("DOC-011")
    elif "legal review" in q and "external user data" in ctx:
        answer = "Legal review is mandatory if the policy exception affects external user data."
        cite_doc("DOC-013")
    elif "rl" in q:
        cite_doc("DOC-010")
        if "post-training" in q and "reinforcement learning" in ctx:
            answer = "In model post-training discussions, RL means Reinforcement Learning."
        elif "only asks" in q or "what should the agent do" in q:
            if policy.abstain_when_low_confidence or policy.answer_style == "synthesis":
                answer = "The agent should ask for clarification because RL can mean Reinforcement Learning or Reliability Label depending on context."
            else:
                answer = "RL means Reinforcement Learning."
        elif policy.abstain_when_low_confidence:
            answer = "RL is ambiguous; it can mean Reinforcement Learning or Reliability Label, so more context is needed."
        else:
            answer = "RL means Reinforcement Learning."
    elif "qwen" in q or "production use" in q:
        if "custom gpu serving" in ctx:
            answer = "Production use requires custom GPU serving, capacity planning, and p95 latency validation."
            cite_doc("DOC-008")
    elif "intern search" in q or "snippets" in q or "acl" in q:
        if "document-level acls" in ctx or "document-level acl" in ctx:
            answer = "Intern Search must enforce document-level ACLs and filter returned snippets by user permissions before an LLM sees them."
            cite_doc("DOC-007")
    elif "current enterprise-search agent" in q or "expensive" in q:
        if "very large frontier models" in ctx:
            answer = (
                "The current enterprise-search agent uses very large frontier models for query planning and answer synthesis, "
                "which makes cost and latency high. The goal is a smaller model with optimized prompts or programs that approaches "
                "the same retrieval and generation quality."
            )
            cite_doc("DOC-012")
            cite_doc("DOC-015")

    if policy.use_grounding_check:
        # If the answer claims a supported fact but has no citation because the relevant doc was not retrieved, force abstention.
        if answer != "I could not find enough supported context to answer." and policy.require_citations and not citations:
            answer = "I found related context but not enough cited evidence to answer confidently."

    if policy.abstain_when_low_confidence and answer == "I could not find enough supported context to answer.":
        answer = "I do not have enough retrieved evidence to answer."

    return _cite(answer, citations, policy)


class SearchAgent:
    def __init__(self, tools: SearchTools, model: ModelProfile, policy: PromptPolicy):
        self.tools = tools
        self.model = model
        self.policy = policy

    def run(self, question: str) -> AgentRun:
        start = perf_counter()
        search_query = _rewrite_query(question) if self.policy.use_query_rewrite else question
        result_sets: list[list[SearchResult]] = []
        tool_calls: list[ToolCall] = []
        kg_claims: list[str] = []

        # Semantic search is the default tool for all variants.
        sem_results = self.tools.semantic_search(search_query, k=self.policy.top_k_semantic)
        result_sets.append(sem_results)
        tool_calls.append(ToolCall("semantic", search_query, self.policy.top_k_semantic, len(sem_results)))

        calls_used = 1
        if self.policy.use_lexical_for_exact and calls_used < self.policy.max_tool_calls and _looks_exact_or_identifier(question):
            lex_results = self.tools.lexical_search(search_query, k=self.policy.top_k_lexical)
            result_sets.append(lex_results)
            tool_calls.append(ToolCall("lexical", search_query, self.policy.top_k_lexical, len(lex_results)))
            calls_used += 1

        if self.policy.use_kg_for_relationships and calls_used < self.policy.max_tool_calls and _looks_relationship_question(question):
            kg_claims = self.tools.kg_claims(search_query)
            tool_calls.append(ToolCall("kg", search_query, None, len(kg_claims)))
            calls_used += 1

        merged = merge_results(result_sets, max_docs=self.policy.context_budget_docs)
        docs = _truncate_docs(merged, max_docs=self.policy.context_budget_docs)
        answer, citations = _local_generate_answer(question, docs, kg_claims, self.policy)

        input_tokens = len(tokenize(question)) + sum(len(tokenize(d.text)) for d in docs) + len(tokenize(self.policy.instruction))
        output_tokens = len(tokenize(answer))
        estimated_cost = (
            input_tokens / 1000.0 * self.model.input_cost_per_1k_tokens
            + output_tokens / 1000.0 * self.model.output_cost_per_1k_tokens
        )
        simulated_latency_ms = (
            self.model.base_latency_ms
            + len(tool_calls) * self.model.latency_per_tool_call_ms
            + (input_tokens + output_tokens) / 1000.0 * self.model.latency_per_1k_tokens_ms
        )
        # Include tiny real elapsed time so repeated runs are realistic but stable enough.
        elapsed_ms = (perf_counter() - start) * 1000.0
        latency_ms = round(simulated_latency_ms + min(elapsed_ms, 10.0), 3)

        return AgentRun(
            question=question,
            answer=answer,
            citations=citations,
            retrieved_doc_ids=tuple(res.doc.doc_id for res in merged),
            tool_calls=tuple(tool_calls),
            kg_claims=tuple(kg_claims),
            policy=self.policy,
            model=self.model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost,
        )
