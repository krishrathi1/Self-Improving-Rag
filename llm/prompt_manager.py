"""
APEX Prompt Manager — Versioned prompt templates with performance tracking
and automatic refinement for underperforming prompts.
Part of the self-improvement loop.
"""

import json
from typing import Optional
from datetime import datetime
from loguru import logger

from llm.llm_layer import get_llm_layer


class PromptVersion:
    """A single version of a prompt template with performance tracking."""

    def __init__(self, template: str, version: str, parent: Optional[str] = None):
        self.template = template
        self.version = version
        self.parent_version = parent
        self.active = True
        self.uses = 0
        self.total_crag_score = 0.0
        self.avg_crag_score = 0.0
        self.created_at = datetime.utcnow()
        self.refinement_reason: Optional[str] = None

    def track(self, crag_score: float):
        """Track a usage with its CRAG score."""
        self.uses += 1
        self.total_crag_score += crag_score
        self.avg_crag_score = self.total_crag_score / self.uses

    @property
    def needs_refinement(self) -> bool:
        """Check if this prompt needs auto-refinement."""
        return self.uses >= 20 and self.avg_crag_score < 0.6


class PromptManager:
    """
    Manages versioned prompt templates with:
    - Performance tracking per version
    - Automatic refinement when avg CRAG score drops below threshold
    - Hot-swap between versions without restart
    """

    def __init__(self):
        self._templates: dict[str, dict[str, PromptVersion]] = {}
        self._active_versions: dict[str, str] = {}
        self._register_defaults()
        logger.info("📝 Prompt Manager initialized with default templates")

    def _register_defaults(self):
        """Register the default prompt templates."""

        # --- GraphRAG QA Prompt ---
        self._register("graphrag_qa", "v1", """You are an expert research assistant with access to a structured knowledge graph. Use ONLY the provided graph context and retrieved passages to answer the question accurately.

**Entity Relationships (from Knowledge Graph):**
{relationships}

**Retrieved Passages:**
{context}

**User Question:** {question}

Instructions:
1. Prioritize information from the entity relationships — these are verified structural facts
2. Use the retrieved passages to add detail and nuance
3. If the context is insufficient, clearly state what information is missing
4. Cite specific entities and relationships when making claims
5. Be concise but thorough — quality over quantity

**Answer:**""")

        # --- Baseline QA Prompt ---
        self._register("baseline_qa", "v1", """Answer the following question using the provided context passages.

Context:
{context}

Question: {question}

Provide a clear, accurate answer based on the context above. If the context doesn't contain enough information, say so.

Answer:""")

        # --- CRAG Grading Prompt ---
        self._register("crag_grading", "v1", """You are an impartial relevance judge. Evaluate whether the retrieved passages contain sufficient information to accurately answer the question.

**Question:** {query}

**Retrieved Passages:**
{chunks}

Rate the overall relevance on a scale from 0.0 to 1.0:
- 0.9-1.0: Passages directly and completely answer the question
- 0.7-0.89: Passages contain most of the needed information
- 0.4-0.69: Passages are partially relevant, answer requires inference
- 0.1-0.39: Passages are tangentially related, mostly insufficient
- 0.0-0.09: Passages are completely irrelevant to the question

Respond with ONLY valid JSON (no markdown):
{{"score": <float>, "reason": "<brief 1-sentence explanation>"}}""")

        # --- Entity Extraction Prompt ---
        self._register("entity_extraction", "v1", """Extract all named entities from the following text. Return them as a JSON array.

Text: {text}

For each entity, identify:
- name: the entity name as mentioned in the text
- entity_type: one of [PERSON, ORGANIZATION, LOCATION, MEDICAL_TERM, DRUG, CONDITION, CONCEPT, OTHER]

Respond with ONLY valid JSON (no markdown):
[{{"name": "...", "entity_type": "..."}}]""")

        # --- Relationship Extraction Prompt ---
        self._register("relationship_extraction", "v1", """Extract relationships between entities from the following text.

Text: {text}

Entities already identified: {entities}

For each relationship found, identify:
- source: the source entity name
- target: the target entity name  
- relation_type: type of relationship (e.g., TREATS, CAUSES, RELATED_TO, PART_OF, INTERACTS_WITH)

Respond with ONLY valid JSON (no markdown):
[{{"source": "...", "target": "...", "relation_type": "..."}}]""")

        # --- Query Decomposition Prompt ---
        self._register("query_decomposition", "v1", """Break down this complex question into 2-4 simpler sub-questions that, when answered together, would fully answer the original question.

Original Question: {query}

Respond with ONLY valid JSON (no markdown):
[{{"sub_query": "...", "purpose": "..."}}]""")

        # --- Accuracy Judge Prompt ---
        self._register("accuracy_judge", "v1", """You are an impartial judge evaluating two answers to the same question.

Question: {query}

Answer A (Baseline LLM):
{answer_a}

Answer B (GraphRAG):
{answer_b}

Rate each answer on these dimensions (0-10 each):
1. **Factual Accuracy**: Are the facts correct and verifiable?
2. **Completeness**: Does it fully address all parts of the question?
3. **Relevance**: Does it stay focused on what was asked?
4. **Conciseness**: Is it appropriately concise without being vague?

Respond with ONLY valid JSON (no markdown):
{{
    "answer_a": {{"accuracy": <int>, "completeness": <int>, "relevance": <int>, "conciseness": <int>, "total": <int>}},
    "answer_b": {{"accuracy": <int>, "completeness": <int>, "relevance": <int>, "conciseness": <int>, "total": <int>}},
    "winner": "A" | "B" | "tie",
    "explanation": "<brief reasoning>"
}}""")

    def _register(self, name: str, version: str, template: str, parent: Optional[str] = None):
        """Register a prompt template version."""
        if name not in self._templates:
            self._templates[name] = {}

        self._templates[name][version] = PromptVersion(template, version, parent)
        self._active_versions[name] = version

    def get(self, name: str, **kwargs) -> str:
        """
        Get a rendered prompt template by name.
        Uses the currently active version.
        """
        if name not in self._templates:
            raise KeyError(f"Unknown prompt template: {name}")

        version = self._active_versions[name]
        pv = self._templates[name][version]
        return pv.template.format(**kwargs)

    def active_version(self, name: str) -> str:
        """Get the active version string for a template."""
        return self._active_versions.get(name, "v1")

    async def track_and_refine(self, template_name: str, version: str, crag_score: float):
        """
        Track prompt performance and trigger auto-refinement if needed.
        This is part of the self-improvement feedback loop.
        """
        if template_name not in self._templates:
            return
        if version not in self._templates[template_name]:
            return

        pv = self._templates[template_name][version]
        pv.track(crag_score)

        logger.debug(
            f"Prompt '{template_name}' {version}: "
            f"avg_score={pv.avg_crag_score:.3f}, uses={pv.uses}"
        )

        if pv.needs_refinement:
            logger.warning(
                f"⚡ Prompt '{template_name}' {version} underperforming "
                f"(avg={pv.avg_crag_score:.3f}). Triggering auto-refinement..."
            )
            await self._auto_refine(template_name, version)

    async def _auto_refine(self, template_name: str, version: str):
        """
        Use the LLM to generate an improved version of an underperforming prompt.
        The old version is deactivated and the new one becomes active.
        """
        pv = self._templates[template_name][version]
        llm = get_llm_layer()

        refinement_prompt = f"""This prompt template has an average quality score of {pv.avg_crag_score:.2f}/1.0 (target: 0.7+) over {pv.uses} uses. Improve it to produce better, more focused responses.

Current template:
---
{pv.template}
---

Generate an improved version that:
1. Provides clearer, more specific instructions
2. Better structures how the context should be used
3. Reduces the chance of hallucination
4. Encourages more precise and relevant answers
5. Maintains all {{variable}} placeholders exactly as they are

Return ONLY the improved template text. Do not include explanations or markdown formatting."""

        try:
            response = await llm.generate(refinement_prompt, temperature=0.3)
            new_version_num = int(version.lstrip("v")) + 1
            new_version = f"v{new_version_num}"

            self._register(
                template_name, new_version, response.text.strip(), parent=version
            )

            # Deactivate old version
            pv.active = False
            self._active_versions[template_name] = new_version

            # Store refinement metadata
            new_pv = self._templates[template_name][new_version]
            new_pv.refinement_reason = (
                f"Auto-refined from {version} (avg_score={pv.avg_crag_score:.3f})"
            )

            logger.success(
                f"✨ Prompt '{template_name}' refined: {version} → {new_version}"
            )

        except Exception as e:
            logger.error(f"Auto-refinement failed for '{template_name}': {e}")

    def get_all_stats(self) -> dict:
        """Get performance stats for all prompt templates."""
        stats = {}
        for name, versions in self._templates.items():
            stats[name] = {
                "active_version": self._active_versions[name],
                "versions": {
                    v: {
                        "uses": pv.uses,
                        "avg_crag_score": round(pv.avg_crag_score, 3),
                        "active": pv.active,
                        "parent": pv.parent_version,
                        "refinement_reason": pv.refinement_reason,
                    }
                    for v, pv in versions.items()
                },
            }
        return stats

    @property
    def total_refinements(self) -> int:
        """Count total auto-refinements performed."""
        count = 0
        for versions in self._templates.values():
            for pv in versions.values():
                if pv.refinement_reason:
                    count += 1
        return count


# Singleton
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get or create the singleton prompt manager."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
