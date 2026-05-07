from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    body: str
    source: str
    acl_group: str = "all"
    entities: tuple[str, ...] = field(default_factory=tuple)

    @property
    def text(self) -> str:
        return f"{self.title}. {self.body}"


@dataclass(frozen=True)
class EvalExample:
    qid: str
    split: str
    category: str
    question: str
    expected_answer: str
    expected_doc_ids: tuple[str, ...]
    must_contain: tuple[str, ...]
    expected_tool_types: tuple[str, ...]
    notes: str = ""


def build_documents() -> list[Document]:
    """Tiny synthetic enterprise corpus inspired by the KARL/SynthWiki setup.

    The content is fictional and safe. The categories mirror the executive summary:
    factual lookup, multi-document synthesis, entity reasoning, procedural/how-to,
    conditional/policy, and absence/ambiguity.
    """

    return [
        Document(
            doc_id="DOC-001",
            title="Ads Privacy Retention Policy",
            source="MetaWiki/Privacy",
            entities=("Ads Logs", "Campaign Logs", "Legal Hold"),
            body=(
                "Campaign logs are retained for 90 days. Training-debug logs are retained for 30 days. "
                "A legal hold overrides normal deletion and can preserve records until the hold is released. "
                "EU data subject request artifacts must be reviewed within 14 days."
            ),
        ),
        Document(
            doc_id="DOC-002",
            title="Budget Transfer Process",
            source="MetaWiki/FinanceOps",
            entities=("FinanceOps", "CFO Approval", "Budget Transfer"),
            body=(
                "To transfer ad budget between business units, open a FinanceOps ticket with source cost center, "
                "destination cost center, amount, and business justification. Transfers under $250k need manager "
                "and finance approval. Transfers over $250k require CFO approval. The standard SLA is two business days."
            ),
        ),
        Document(
            doc_id="DOC-003",
            title="Project Falcon Infrastructure Brief",
            source="MetaWiki/Infra",
            entities=("Project Falcon", "Infra Reliability", "Ashburn", "Prineville"),
            body=(
                "Project Falcon is owned by the Infra Reliability team. It supports GPU capacity planning and failover "
                "for the Ashburn and Prineville data center regions. The project is not owned by FinanceOps."
            ),
        ),
        Document(
            doc_id="DOC-004",
            title="EPKG Overview",
            source="MetaWiki/KnowledgeGraph",
            entities=("EPKG", "Entities", "Relations", "Claims", "Summaries"),
            body=(
                "The Enterprise Products Knowledge Graph, or EPKG, exposes structured tools for entities, relations, "
                "claims, and summaries. It is useful when an agent must reason over relationships rather than simply "
                "retrieve a paragraph."
            ),
        ),
        Document(
            doc_id="DOC-005",
            title="Search Evaluation Metrics",
            source="MetaWiki/SearchEval",
            entities=("Recall@k", "MRR", "Groundedness", "Latency", "Tool Calls"),
            body=(
                "Search-agent evaluations should report retrieval Recall@5, Recall@10, Mean Reciprocal Rank, precision, "
                "generation correctness, completeness, groundedness, average latency per query, and average tool calls per query."
            ),
        ),
        Document(
            doc_id="DOC-006",
            title="Enterprise Search Tool Guide",
            source="MetaWiki/SearchTools",
            entities=("Semantic Search", "Lexical Search", "KG Tools"),
            body=(
                "Semantic search is best for conceptual or paraphrased queries. Lexical search is best for exact names, "
                "IDs, acronyms, policy codes, and numeric thresholds. Knowledge-graph tools are best for entity relationships, "
                "ownership, claims, and summaries."
            ),
        ),
        Document(
            doc_id="DOC-007",
            title="Access Control for Intern Search",
            source="MetaWiki/Security",
            entities=("Intern Search", "ACL", "Permissions"),
            body=(
                "Intern Search must enforce document-level ACLs. Returned snippets must be filtered by the requesting user's "
                "permissions before an LLM sees them. Prompts, traces, and model outputs may be logged in a dev workspace only "
                "when approved by the engagement owner."
            ),
        ),
        Document(
            doc_id="DOC-008",
            title="Model Serving Candidate Notes",
            source="MetaWiki/AIPlatform",
            entities=("Qwen", "Custom GPU Serving", "Meta Model"),
            body=(
                "Qwen 3.5 110B is an experimental candidate for smaller-model retrieval and generation. A Meta-created model "
                "may also be evaluated. Production use requires custom GPU serving, capacity planning, and p95 latency validation."
            ),
        ),
        Document(
            doc_id="DOC-009",
            title="Search Latency Incident Procedure",
            source="MetaWiki/SRE",
            entities=("SEV2", "Search Reliability", "P95"),
            body=(
                "If search p95 latency exceeds 8 seconds for 10 consecutive minutes, file a SEV2 in the Search Reliability queue. "
                "Page the search on-call only if user-facing traffic is affected or error rate exceeds 2%."
            ),
        ),
        Document(
            doc_id="DOC-010",
            title="Acronym Disambiguation Guide",
            source="MetaWiki/Glossary",
            entities=("RL", "Reinforcement Learning", "Reliability Label"),
            body=(
                "RL can mean Reinforcement Learning in model post-training discussions. RL can also mean Reliability Label in "
                "search-quality dashboards. The agent should use surrounding context and ask for clarification when context is missing."
            ),
        ),
        Document(
            doc_id="DOC-011",
            title="Wellness Reimbursement Policy",
            source="MetaWiki/PeopleOps",
            entities=("Wellness", "Gym", "Ergonomic Furniture"),
            body=(
                "The annual wellness reimbursement maximum is $750. Gym memberships and fitness classes are eligible. Ergonomic "
                "furniture is handled under the workplace equipment program, not the wellness program. The policy does not mention yoga mats."
            ),
        ),
        Document(
            doc_id="DOC-012",
            title="Agentic Retrieval Application Baseline",
            source="MetaWiki/AgentPlatform",
            entities=("Opus", "GPT-5.5", "Enterprise Search Agent"),
            body=(
                "The current enterprise-search agent uses very large frontier models for query planning and answer synthesis. "
                "Quality is acceptable on many examples, but cost and latency are high. The goal is to evaluate whether a smaller "
                "model with optimized prompts or programs can approach the same retrieval and generation quality."
            ),
        ),
        Document(
            doc_id="DOC-013",
            title="Policy Exception Workflow",
            source="MetaWiki/PolicyOps",
            entities=("Policy Exception", "VP Approval", "Legal Review"),
            body=(
                "Policy exceptions require written business justification. If the exception affects external user data, legal review "
                "is mandatory. If the exception affects more than one region, VP approval is also required."
            ),
        ),
        Document(
            doc_id="DOC-014",
            title="Knowledge Graph Claims for Project Falcon",
            source="EPKG/Claims",
            entities=("Project Falcon", "Infra Reliability", "GPU Capacity"),
            body=(
                "Claim: Project Falcon owned_by Infra Reliability. Claim: Project Falcon supports GPU Capacity Planning. "
                "Claim: Project Falcon deployed_in Ashburn. Claim: Project Falcon deployed_in Prineville."
            ),
        ),
        Document(
            doc_id="DOC-015",
            title="Retrieval Prompting Tips",
            source="MetaWiki/SearchPlaybook",
            entities=("Query Rewrite", "Citations", "Abstention"),
            body=(
                "For multi-hop questions, rewrite the user request into smaller search queries. Always cite document IDs when "
                "answering from enterprise context. If the retrieved context does not support the answer, say what is missing instead "
                "of guessing."
            ),
        ),
    ]


def build_eval_examples() -> list[EvalExample]:
    return [
        EvalExample(
            qid="Q001",
            split="train",
            category="factual_lookup",
            question="How long are campaign logs retained?",
            expected_answer="Campaign logs are retained for 90 days.",
            expected_doc_ids=("DOC-001",),
            must_contain=("90", "days"),
            expected_tool_types=("semantic",),
        ),
        EvalExample(
            qid="Q002",
            split="train",
            category="conditional_policy",
            question="What approval is required for a budget transfer over $250k?",
            expected_answer="Transfers over $250k require CFO approval.",
            expected_doc_ids=("DOC-002",),
            must_contain=("CFO", "approval"),
            expected_tool_types=("semantic", "lexical"),
        ),
        EvalExample(
            qid="Q003",
            split="train",
            category="entity_relationship",
            question="Who owns Project Falcon and which regions does it support?",
            expected_answer="Project Falcon is owned by Infra Reliability and supports Ashburn and Prineville.",
            expected_doc_ids=("DOC-003", "DOC-014"),
            must_contain=("Infra Reliability", "Ashburn", "Prineville"),
            expected_tool_types=("semantic", "kg"),
        ),
        EvalExample(
            qid="Q004",
            split="train",
            category="procedural_how_to",
            question="How should I report a search latency incident if p95 is above 8 seconds for 10 minutes?",
            expected_answer="File a SEV2 in the Search Reliability queue; page on-call only if user-facing traffic is affected or error rate exceeds 2%.",
            expected_doc_ids=("DOC-009",),
            must_contain=("SEV2", "Search Reliability", "8 seconds"),
            expected_tool_types=("semantic", "lexical"),
        ),
        EvalExample(
            qid="Q005",
            split="train",
            category="multi_document_synthesis",
            question="Which tool should an agent use for relationship claims, and which metrics should evaluate the search agent?",
            expected_answer="Use EPKG/KG tools for relationship claims; evaluate with Recall@5/10, MRR, precision, correctness, completeness, groundedness, latency, and average tool calls.",
            expected_doc_ids=("DOC-004", "DOC-005", "DOC-006"),
            must_contain=("KG", "claims", "Recall", "MRR", "groundedness", "latency"),
            expected_tool_types=("semantic", "kg"),
        ),
        EvalExample(
            qid="Q006",
            split="train",
            category="absence_ambiguity_scope",
            question="Does the wellness policy explicitly mention reimbursement for yoga mats?",
            expected_answer="No. The policy does not explicitly mention yoga mats.",
            expected_doc_ids=("DOC-011",),
            must_contain=("does not", "mention", "yoga mats"),
            expected_tool_types=("semantic", "lexical"),
        ),
        EvalExample(
            qid="Q007",
            split="train",
            category="factual_lookup",
            question="What is the annual wellness reimbursement maximum?",
            expected_answer="The annual wellness reimbursement maximum is $750.",
            expected_doc_ids=("DOC-011",),
            must_contain=("750",),
            expected_tool_types=("semantic",),
        ),
        EvalExample(
            qid="Q008",
            split="train",
            category="conditional_policy",
            question="When is legal review mandatory for a policy exception?",
            expected_answer="Legal review is mandatory if the exception affects external user data.",
            expected_doc_ids=("DOC-013",),
            must_contain=("legal review", "external user data"),
            expected_tool_types=("semantic",),
        ),
        EvalExample(
            qid="Q009",
            split="dev",
            category="ambiguous_acronym",
            question="In model post-training discussions, what does RL mean?",
            expected_answer="In model post-training discussions, RL means Reinforcement Learning.",
            expected_doc_ids=("DOC-010",),
            must_contain=("Reinforcement Learning",),
            expected_tool_types=("semantic", "lexical"),
        ),
        EvalExample(
            qid="Q010",
            split="dev",
            category="operational_feasibility",
            question="What must be validated before production use of Qwen 3.5 110B for search?",
            expected_answer="Production use requires custom GPU serving, capacity planning, and p95 latency validation.",
            expected_doc_ids=("DOC-008",),
            must_contain=("custom GPU serving", "capacity planning", "p95 latency"),
            expected_tool_types=("semantic", "lexical"),
        ),
        EvalExample(
            qid="Q011",
            split="dev",
            category="security_acl",
            question="What must Intern Search enforce before LLMs see returned snippets?",
            expected_answer="Intern Search must enforce document-level ACLs and filter snippets by user permissions before an LLM sees them.",
            expected_doc_ids=("DOC-007",),
            must_contain=("ACL", "permissions", "before an LLM"),
            expected_tool_types=("semantic", "lexical"),
        ),
        EvalExample(
            qid="Q012",
            split="dev",
            category="multi_document_synthesis",
            question="Why is the current enterprise-search agent expensive, and what is the proposed smaller-model goal?",
            expected_answer="It uses very large frontier models for query planning and answer synthesis, so cost and latency are high; the goal is a smaller model with optimized prompts or programs that approaches the same retrieval and generation quality.",
            expected_doc_ids=("DOC-012", "DOC-015"),
            must_contain=("large", "cost", "latency", "smaller", "optimized"),
            expected_tool_types=("semantic",),
        ),
        EvalExample(
            qid="Q013",
            split="test",
            category="factual_lookup",
            question="How long are training-debug logs retained?",
            expected_answer="Training-debug logs are retained for 30 days.",
            expected_doc_ids=("DOC-001",),
            must_contain=("30", "days"),
            expected_tool_types=("semantic",),
        ),
        EvalExample(
            qid="Q014",
            split="test",
            category="conditional_policy",
            question="Which budget transfers require CFO approval?",
            expected_answer="Budget transfers over $250k require CFO approval.",
            expected_doc_ids=("DOC-002",),
            must_contain=("over", "250k", "CFO"),
            expected_tool_types=("semantic", "lexical"),
        ),
        EvalExample(
            qid="Q015",
            split="test",
            category="entity_relationship",
            question="Which team owns Falcon and what does Falcon support?",
            expected_answer="Project Falcon is owned by Infra Reliability and supports GPU capacity planning and failover.",
            expected_doc_ids=("DOC-003", "DOC-014"),
            must_contain=("Infra Reliability", "GPU capacity"),
            expected_tool_types=("semantic", "kg"),
        ),
        EvalExample(
            qid="Q016",
            split="test",
            category="procedural_how_to",
            question="When should the search on-call be paged during a latency incident?",
            expected_answer="Page the search on-call only if user-facing traffic is affected or error rate exceeds 2%.",
            expected_doc_ids=("DOC-009",),
            must_contain=("user-facing traffic", "2%"),
            expected_tool_types=("semantic", "lexical"),
        ),
        EvalExample(
            qid="Q017",
            split="test",
            category="absence_ambiguity_scope",
            question="If someone only asks what RL means, what should the agent do?",
            expected_answer="The agent should use context or ask for clarification because RL can mean Reinforcement Learning or Reliability Label.",
            expected_doc_ids=("DOC-010",),
            must_contain=("ask for clarification", "Reinforcement Learning", "Reliability Label"),
            expected_tool_types=("semantic", "lexical"),
        ),
        EvalExample(
            qid="Q018",
            split="test",
            category="multi_document_synthesis",
            question="For exact policy codes and ownership claims, which tools should the search agent prefer?",
            expected_answer="Use lexical search for exact policy codes and KG/EPKG tools for ownership claims or relationship reasoning.",
            expected_doc_ids=("DOC-006", "DOC-004"),
            must_contain=("lexical", "exact", "KG", "ownership"),
            expected_tool_types=("semantic", "lexical", "kg"),
        ),
    ]


def by_split(examples: Iterable[EvalExample], split: str) -> list[EvalExample]:
    return [ex for ex in examples if ex.split == split]
