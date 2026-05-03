"""Fetch Wikipedia article revision history via the MediaWiki API.

Retrieves the full revision history for a set of articles, including
timestamps and content for each revision. Content is used to extract
facts and track their persistence across revisions.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TemporalDecayResearch/1.0 (academic research)"

# Rate limiting
MIN_REQUEST_INTERVAL = 1.0  # seconds between API requests


@dataclass(frozen=True)
class Revision:
    """A single Wikipedia revision."""

    revid: int
    timestamp: str  # ISO 8601
    size: int
    content: str | None = None  # wikitext content (if fetched)


@dataclass
class ArticleRevisions:
    """All revisions for a single article."""

    title: str
    pageid: int
    revisions: list[Revision]


def fetch_revision_list(
    title: str,
    limit: int = 500,
    start_date: str | None = None,
    end_date: str | None = None,
) -> ArticleRevisions:
    """Fetch revision metadata (no content) for an article.

    Args:
        title: Wikipedia article title
        limit: max revisions to fetch (API max is 500 per request)
        start_date: ISO timestamp, fetch revisions from this date (newest first by default)
        end_date: ISO timestamp, fetch revisions until this date
    """
    revisions = []
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "ids|timestamp|size",
        "rvlimit": str(min(limit, 500)),
        "rvdir": "newer",  # oldest first
        "format": "json",
    }
    if start_date:
        params["rvstart"] = start_date
    if end_date:
        params["rvend"] = end_date

    pageid = None

    while True:
        resp = requests.get(
            API_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        for pid, page_data in pages.items():
            pageid = int(pid)
            for rev in page_data.get("revisions", []):
                revisions.append(Revision(
                    revid=rev["revid"],
                    timestamp=rev["timestamp"],
                    size=rev.get("size", 0),
                ))

        if len(revisions) >= limit:
            break

        # Handle continuation
        cont = data.get("continue")
        if cont:
            params.update(cont)
            time.sleep(MIN_REQUEST_INTERVAL)
        else:
            break

    return ArticleRevisions(
        title=title,
        pageid=pageid or 0,
        revisions=revisions[:limit],
    )


def fetch_revision_content(
    revids: list[int],
    batch_size: int = 50,
) -> dict[int, str]:
    """Fetch wikitext content for specific revision IDs.

    Args:
        revids: list of revision IDs to fetch content for
        batch_size: number of revisions per API request (max 50)

    Returns:
        dict mapping revid -> wikitext content
    """
    contents = {}

    for i in range(0, len(revids), batch_size):
        batch = revids[i:i + batch_size]
        params = {
            "action": "query",
            "revids": "|".join(str(r) for r in batch),
            "prop": "revisions",
            "rvprop": "ids|content",
            "rvslots": "main",
            "format": "json",
        }

        resp = requests.get(
            API_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        for pid, page_data in pages.items():
            for rev in page_data.get("revisions", []):
                revid = rev["revid"]
                content = rev.get("slots", {}).get("main", {}).get("*", "")
                contents[revid] = content

        if i + batch_size < len(revids):
            time.sleep(MIN_REQUEST_INTERVAL)

    return contents


def sample_revisions(
    article_revs: ArticleRevisions,
    max_revisions: int = 50,
    strategy: str = "uniform",
) -> list[int]:
    """Sample revision IDs from the full list for content fetching.

    We don't need every revision -- sampling at regular intervals
    captures the temporal dynamics while staying within API limits.

    Args:
        article_revs: full revision list
        max_revisions: max number of revisions to sample
        strategy: "uniform" for evenly spaced, "all" for everything
    """
    revids = [r.revid for r in article_revs.revisions]

    if strategy == "all" or len(revids) <= max_revisions:
        return revids

    if strategy == "uniform":
        indices = [int(i) for i in range(0, len(revids), max(1, len(revids) // max_revisions))]
        # Always include first and last
        if 0 not in indices:
            indices = [0] + indices
        if len(revids) - 1 not in indices:
            indices.append(len(revids) - 1)
        return [revids[i] for i in indices[:max_revisions]]

    return revids[:max_revisions]


# --- Curated article list for the experiment ---

# Articles chosen to span different temporal dynamics:
# - Stable scientific facts (genetics, physics)
# - Evolving medical knowledge (treatments, guidelines)
# - Current events / rapidly changing topics
# - Biographical facts (mix of permanent and changing)

EXPERIMENT_ARTICLES = [
    # === Stable science (permanent facts) ===
    "BRAF (gene)", "TP53", "DNA", "Speed of light", "Periodic table",
    "EGFR", "BRCA1", "Hemoglobin", "Mitochondrion", "Photosynthesis",
    "General relativity", "Quantum mechanics", "Electron", "Proton",
    "Planck constant", "Avogadro constant", "Water", "Carbon",
    # === Evolving medical knowledge ===
    "Melanoma", "Immunotherapy", "COVID-19", "Pembrolizumab",
    "CRISPR gene editing", "Breast cancer", "Lung cancer",
    "Type 2 diabetes", "Hypertension", "Alzheimer's disease",
    "Influenza", "HIV/AIDS", "Malaria", "Tuberculosis",
    "Aspirin", "Metformin", "Ibuprofen", "Paracetamol",
    "Monoclonal antibody", "Gene therapy",
    # === Technology (fast-evolving) ===
    "Bitcoin", "Artificial intelligence", "ChatGPT",
    "Machine learning", "Deep learning", "Blockchain",
    "5G", "Smartphone", "Electric vehicle", "Self-driving car",
    "SpaceX", "Tesla, Inc.", "Google", "Apple Inc.",
    "Microsoft", "Meta Platforms", "Netflix", "Twitter",
    # === Current events / politics (volatile) ===
    "Climate change", "European Union", "United Nations",
    "Russia", "Ukraine", "NATO", "World War II",
    "2024 United States presidential election",
    "Syrian civil war", "Israeli–Palestinian conflict",
    # === Biographies (permanent + changing) ===
    "Barack Obama", "Albert Einstein", "Elon Musk",
    "Marie Curie", "Ada Lovelace", "Isaac Newton",
    "Charles Darwin", "Nikola Tesla", "Alan Turing",
    "Nelson Mandela", "Mahatma Gandhi", "Cleopatra",
    "Leonardo da Vinci", "William Shakespeare",
    # === Geography / institutions (mostly stable) ===
    "Tokyo", "Harvard University", "Amazon (company)",
    "World Health Organization", "International Space Station",
    "New York City", "London", "Paris", "Beijing",
    "Mount Everest", "Pacific Ocean", "Sahara",
    "Oxford University", "MIT", "Stanford University",
    # === History (permanent but revised interpretations) ===
    "Roman Empire", "French Revolution", "Industrial Revolution",
    "Cold War", "Renaissance", "Ancient Egypt",
    # === Culture / sports (mix) ===
    "Olympic Games", "FIFA World Cup", "Beatles",
    "Star Wars", "Harry Potter", "Super Bowl",
]


def save_revisions(article_revs: ArticleRevisions, output_dir: Path) -> None:
    """Save fetched revision data to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_title = article_revs.title.replace("/", "_").replace(" ", "_")
    filepath = output_dir / f"{safe_title}.json"

    data = {
        "title": article_revs.title,
        "pageid": article_revs.pageid,
        "revisions": [
            {
                "revid": r.revid,
                "timestamp": r.timestamp,
                "size": r.size,
                "content": r.content,
            }
            for r in article_revs.revisions
        ],
    }

    with open(filepath, "w") as f:
        json.dump(data, f)

    logger.info(f"Saved {len(article_revs.revisions)} revisions for '{article_revs.title}' to {filepath}")


def load_revisions(filepath: Path) -> ArticleRevisions:
    """Load revision data from disk."""
    with open(filepath) as f:
        data = json.load(f)

    return ArticleRevisions(
        title=data["title"],
        pageid=data["pageid"],
        revisions=[
            Revision(
                revid=r["revid"],
                timestamp=r["timestamp"],
                size=r["size"],
                content=r.get("content"),
            )
            for r in data["revisions"]
        ],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick test: fetch revision list for one article
    print("Fetching revision list for 'BRAF (gene)'...")
    revs = fetch_revision_list("BRAF (gene)", limit=20)
    print(f"  Found {len(revs.revisions)} revisions")
    for r in revs.revisions[:5]:
        print(f"    {r.timestamp} (revid={r.revid}, size={r.size})")

    # Sample and fetch content for a few
    sampled = sample_revisions(revs, max_revisions=3)
    print(f"\n  Fetching content for {len(sampled)} sampled revisions...")
    contents = fetch_revision_content(sampled)
    for revid, content in contents.items():
        print(f"    revid={revid}: {len(content)} chars")
