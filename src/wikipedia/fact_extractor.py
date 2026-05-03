"""Extract structured facts from Wikipedia wikitext and track across revisions.

Uses regex-based extraction for infobox fields and section-level content,
producing a temporal KG where each fact has a first-seen and last-seen timestamp.

For the paper's purposes, we need to identify:
- Facts that persist across many revisions (permanent)
- Facts that change between revisions (volatile)
- The velocity (revision frequency) and volatility (change magnitude) of each fact
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WikiFact:
    """A single extracted fact from a Wikipedia revision."""

    article: str
    predicate: str  # the field or section name
    value: str  # the extracted value
    fact_key: str  # unique key: article + predicate (the concept)


@dataclass
class FactObservation:
    """A fact observed at a specific revision."""

    fact: WikiFact
    revision_timestamp: str
    revision_id: int


@dataclass
class FactTimeline:
    """The temporal history of a single concept (article, predicate) across revisions."""

    article: str
    predicate: str
    observations: list[tuple[str, str, int]] = field(default_factory=list)
    # Each observation: (timestamp, value, revid)

    @property
    def fact_key(self) -> str:
        return f"{self.article}::{self.predicate}"

    @property
    def n_observations(self) -> int:
        return len(self.observations)

    @property
    def values(self) -> list[str]:
        return [v for _, v, _ in self.observations]

    @property
    def timestamps(self) -> list[str]:
        return [t for t, _, _ in self.observations]

    @property
    def n_value_changes(self) -> int:
        """Count how many times the value changed between consecutive observations."""
        changes = 0
        for i in range(1, len(self.observations)):
            if self.observations[i][1] != self.observations[i - 1][1]:
                changes += 1
        return changes

    @property
    def volatility_ratio(self) -> float:
        """Fraction of consecutive observations where value changed."""
        if self.n_observations < 2:
            return 0.0
        return self.n_value_changes / (self.n_observations - 1)

    @property
    def is_stable(self) -> bool:
        """True if value never changed across all observations."""
        return self.n_value_changes == 0

    @property
    def timespan_days(self) -> float:
        """Days between first and last observation."""
        if self.n_observations < 2:
            return 0.0
        t_first = datetime.fromisoformat(self.timestamps[0].replace("Z", "+00:00"))
        t_last = datetime.fromisoformat(self.timestamps[-1].replace("Z", "+00:00"))
        return (t_last - t_first).total_seconds() / 86400

    @property
    def velocity(self) -> float:
        """Observations per day."""
        span = self.timespan_days
        if span <= 0:
            return 0.0
        return self.n_observations / span


def extract_infobox_fields(wikitext: str) -> dict[str, str]:
    """Extract key-value pairs from Wikipedia infobox-style templates.

    Handles various template formats:
    - {{Infobox gene | ... }}
    - {{GNF_Protein_box | ... }}
    - {{Drugbox | ... }}
    - {{Infobox person | ... }}
    etc.
    """
    fields = {}

    # Match any template that contains key=value pairs (infobox-like)
    # Common patterns: Infobox*, *_box, Drugbox, Taxobox, etc.
    infobox_pattern = re.compile(
        r'\{\{(?:[Ii]nfobox[^}]*?|[A-Z]\w*_?[Bb]ox|[Dd]rugbox|[Tt]axobox)[^}]*?\n(.*?)\}\}',
        re.DOTALL,
    )
    match = infobox_pattern.search(wikitext)
    if not match:
        return fields

    content = match.group(1)

    # Extract individual fields
    field_pattern = re.compile(r'\|\s*(\w[\w\s]*?)\s*=\s*(.*?)(?=\n\s*\||$)', re.DOTALL)
    for m in field_pattern.finditer(content):
        key = m.group(1).strip().lower().replace(" ", "_")
        value = m.group(2).strip()
        # Clean wiki markup from value
        value = clean_wiki_markup(value)
        if value and len(value) < 500:  # skip very long values
            fields[key] = value

    return fields


def extract_section_summaries(wikitext: str) -> dict[str, str]:
    """Extract the first paragraph of each section as a fact.

    Sections represent major content areas. The first sentence/paragraph
    captures the key claim of that section.
    """
    sections = {}

    # Split on section headers
    section_pattern = re.compile(r'^(={2,})\s*(.+?)\s*\1', re.MULTILINE)
    matches = list(section_pattern.finditer(wikitext))

    for i, match in enumerate(matches):
        section_name = match.group(2).strip().lower().replace(" ", "_")
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(wikitext)

        content = wikitext[start:end].strip()
        # Take first paragraph (up to double newline or 300 chars)
        first_para = content.split("\n\n")[0][:300]
        first_para = clean_wiki_markup(first_para).strip()

        if first_para and len(first_para) > 20:
            sections[f"section:{section_name}"] = first_para

    return sections


def extract_categories(wikitext: str) -> list[str]:
    """Extract Wikipedia categories."""
    cat_pattern = re.compile(r'\[\[Category:(.+?)(?:\|.*)?\]\]')
    return [m.group(1).strip() for m in cat_pattern.finditer(wikitext)]


def clean_wiki_markup(text: str) -> str:
    """Remove common wiki markup from text."""
    # Remove references
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*/>', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove wiki links, keep text: [[Link|Text]] -> Text, [[Link]] -> Link
    text = re.sub(r'\[\[[^|\]]*\|([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    # Remove templates (simple ones)
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    # Remove bold/italic
    text = re.sub(r"'{2,}", '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_lead_sentence(wikitext: str) -> str | None:
    """Extract the first sentence of the article (before any section header)."""
    # Get content before first section header
    first_header = re.search(r'^==', wikitext, re.MULTILINE)
    if first_header:
        lead = wikitext[:first_header.start()]
    else:
        lead = wikitext[:1000]

    lead = clean_wiki_markup(lead).strip()

    # Take first sentence
    sentences = re.split(r'(?<=[.!?])\s+', lead)
    if sentences and len(sentences[0]) > 10:
        return sentences[0][:300]
    return None


def extract_facts_from_revision(
    article_title: str,
    wikitext: str,
) -> list[WikiFact]:
    """Extract all facts from a single revision's wikitext."""
    facts = []

    # 1. Lead sentence
    lead = extract_lead_sentence(wikitext)
    if lead:
        facts.append(WikiFact(article_title, "lead_sentence", lead,
                              f"{article_title}::lead_sentence"))

    # 2. Infobox fields
    for key, value in extract_infobox_fields(wikitext).items():
        facts.append(WikiFact(article_title, f"infobox:{key}", value,
                              f"{article_title}::infobox:{key}"))

    # 3. Section summaries
    for key, value in extract_section_summaries(wikitext).items():
        facts.append(WikiFact(article_title, key, value,
                              f"{article_title}::{key}"))

    # 4. Categories
    for cat in extract_categories(wikitext):
        facts.append(WikiFact(article_title, f"category:{cat}", cat,
                              f"{article_title}::category:{cat}"))

    return facts


def build_fact_timelines(
    article_title: str,
    revisions: list[dict],
) -> dict[str, FactTimeline]:
    """Build temporal fact timelines from a series of revisions.

    Args:
        article_title: the article name
        revisions: list of dicts with 'timestamp', 'revid', 'content' keys

    Returns:
        dict mapping fact_key -> FactTimeline
    """
    timelines: dict[str, FactTimeline] = {}

    for rev in revisions:
        content = rev.get("content")
        if not content:
            continue

        timestamp = rev["timestamp"]
        revid = rev["revid"]

        facts = extract_facts_from_revision(article_title, content)

        # Track which facts we saw in this revision
        seen_keys = set()

        for fact in facts:
            key = fact.fact_key
            seen_keys.add(key)

            if key not in timelines:
                timelines[key] = FactTimeline(
                    article=article_title,
                    predicate=fact.predicate,
                )

            timelines[key].observations.append((timestamp, fact.value, revid))

    return timelines


def timelines_to_temporal_kg(
    all_timelines: dict[str, FactTimeline],
) -> list[dict]:
    """Convert fact timelines into temporal KG edges for the decay model.

    Each value-change creates a supersession event.
    Each same-value re-observation is a reinforcement.
    """
    edges = []
    edge_id = 0

    for fact_key, timeline in all_timelines.items():
        if timeline.n_observations < 2:
            continue

        current_value = None
        current_edge = None
        current_start = None

        for timestamp, value, revid in timeline.observations:
            if current_value is None:
                # First observation
                current_value = value
                current_start = timestamp
                current_edge = {
                    "edge_id": edge_id,
                    "subject": timeline.article,
                    "predicate": timeline.predicate,
                    "object_value": value,
                    "timestamp": timestamp,
                    "reinforcement_times": [],
                    "superseded_at": None,
                    "velocity": timeline.velocity,
                    "volatility_ratio": timeline.volatility_ratio,
                }
                edge_id += 1
            elif value == current_value:
                # Same value -> reinforcement
                current_edge["reinforcement_times"].append(timestamp)
            else:
                # Different value -> supersession
                current_edge["superseded_at"] = timestamp
                edges.append(current_edge)

                # New edge with new value
                current_value = value
                current_start = timestamp
                current_edge = {
                    "edge_id": edge_id,
                    "subject": timeline.article,
                    "predicate": timeline.predicate,
                    "object_value": value,
                    "timestamp": timestamp,
                    "reinforcement_times": [],
                    "superseded_at": None,
                    "velocity": timeline.velocity,
                    "volatility_ratio": timeline.volatility_ratio,
                }
                edge_id += 1

        # Last edge is right-censored
        if current_edge is not None:
            edges.append(current_edge)

    return edges


if __name__ == "__main__":
    # Quick test with sample wikitext
    sample_wikitext = """
'''BRAF''' is a human [[gene]] that encodes a [[protein]] called B-Raf.

{{Infobox gene
| Name = BRAF
| Symbol = BRAF
| EntrezGene = 673
| OMIM = 164757
| Chromosome = 7
| Arm = q
| Band = 34
}}

== Function ==
The BRAF gene provides instructions for making a protein that helps transmit chemical signals from outside the cell to the cell's nucleus.

== Clinical significance ==
Mutations in BRAF are associated with various cancers, most notably melanoma. The V600E mutation is the most common.

== History ==
BRAF was first identified in 2002 as a human oncogene.

[[Category:Genes on human chromosome 7]]
[[Category:Oncogenes]]
"""

    facts = extract_facts_from_revision("BRAF (gene)", sample_wikitext)
    print(f"Extracted {len(facts)} facts from sample wikitext:")
    for fact in facts:
        print(f"  [{fact.predicate}] = {fact.value[:80]}...")
