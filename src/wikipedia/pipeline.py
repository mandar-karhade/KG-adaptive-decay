"""Wikipedia temporal KG construction pipeline.

Orchestrates: fetch revisions -> extract facts -> build timelines -> create temporal KG.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.wikipedia.revision_fetcher import (
    fetch_revision_list,
    fetch_revision_content,
    sample_revisions,
    save_revisions,
    load_revisions,
    ArticleRevisions,
    Revision,
    EXPERIMENT_ARTICLES,
)
from src.wikipedia.fact_extractor import (
    build_fact_timelines,
    timelines_to_temporal_kg,
    FactTimeline,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/wikipedia")


def fetch_article_data(
    articles: list[str] | None = None,
    max_revisions_per_article: int = 100,
    sample_for_content: int = 40,
    force_refetch: bool = False,
) -> list[ArticleRevisions]:
    """Fetch revision data for a list of articles.

    Caches to disk to avoid re-fetching on subsequent runs.
    """
    if articles is None:
        articles = EXPERIMENT_ARTICLES

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for i, title in enumerate(articles):
        safe_title = title.replace("/", "_").replace(" ", "_")
        cache_path = DATA_DIR / f"{safe_title}.json"

        if cache_path.exists() and not force_refetch:
            logger.info(f"[{i+1}/{len(articles)}] Loading cached: {title}")
            article_revs = load_revisions(cache_path)
        else:
            logger.info(f"[{i+1}/{len(articles)}] Fetching: {title}")

            # Get revision list
            article_revs = fetch_revision_list(title, limit=max_revisions_per_article)
            logger.info(f"  Found {len(article_revs.revisions)} revisions")

            if not article_revs.revisions:
                logger.warning(f"  No revisions found for {title}, skipping")
                continue

            # Sample revisions for content fetching
            sampled_ids = sample_revisions(article_revs, max_revisions=sample_for_content)
            logger.info(f"  Fetching content for {len(sampled_ids)} sampled revisions")

            # Fetch content
            contents = fetch_revision_content(sampled_ids)

            # Merge content into revision objects
            enriched_revisions = []
            for rev in article_revs.revisions:
                content = contents.get(rev.revid)
                enriched_revisions.append(Revision(
                    revid=rev.revid,
                    timestamp=rev.timestamp,
                    size=rev.size,
                    content=content,
                ))

            article_revs = ArticleRevisions(
                title=article_revs.title,
                pageid=article_revs.pageid,
                revisions=enriched_revisions,
            )

            # Cache to disk
            save_revisions(article_revs, DATA_DIR)
            time.sleep(1)  # rate limiting

        results.append(article_revs)

    return results


def build_wikipedia_temporal_kg(
    article_data: list[ArticleRevisions],
) -> pd.DataFrame:
    """Build a temporal KG from fetched Wikipedia article revisions.

    Returns a DataFrame compatible with the synthetic KG format.
    """
    all_timelines: dict[str, FactTimeline] = {}

    for article_revs in article_data:
        # Build revision dicts for the fact extractor
        revisions = [
            {
                "timestamp": r.timestamp,
                "revid": r.revid,
                "content": r.content,
            }
            for r in article_revs.revisions
            if r.content is not None
        ]

        if not revisions:
            continue

        timelines = build_fact_timelines(article_revs.title, revisions)
        all_timelines.update(timelines)

    logger.info(f"Built {len(all_timelines)} fact timelines from {len(article_data)} articles")

    # Convert to temporal KG edges
    raw_edges = timelines_to_temporal_kg(all_timelines)
    logger.info(f"Created {len(raw_edges)} temporal KG edges")

    if not raw_edges:
        return pd.DataFrame()

    # Convert to DataFrame matching synthetic format
    rows = []
    for edge in raw_edges:
        ts = edge["timestamp"]
        # Parse ISO timestamp to days since epoch
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            timestamp_days = (dt - datetime(2000, 1, 1, tzinfo=dt.tzinfo)).total_seconds() / 86400
        except (ValueError, TypeError):
            continue

        sup_at = edge.get("superseded_at")
        sup_days = None
        if sup_at:
            try:
                sup_dt = datetime.fromisoformat(sup_at.replace("Z", "+00:00"))
                sup_days = (sup_dt - datetime(2000, 1, 1, tzinfo=sup_dt.tzinfo)).total_seconds() / 86400
            except (ValueError, TypeError):
                pass

        lifetime = None
        is_censored = True
        if sup_days is not None:
            lifetime = sup_days - timestamp_days
            is_censored = False
        else:
            # Right-censored: use time from creation to now
            now_days = (datetime.now().astimezone() - datetime(2000, 1, 1, tzinfo=dt.tzinfo)).total_seconds() / 86400
            lifetime = now_days - timestamp_days

        if lifetime is not None and lifetime <= 0:
            continue

        rows.append({
            "edge_id": edge["edge_id"],
            "subject": edge["subject"],
            "predicate": edge["predicate"],
            "object_value": edge["object_value"][:200],  # truncate long values
            "timestamp": timestamp_days,
            "velocity": edge.get("velocity", 0.0),
            "volatility": edge.get("volatility_ratio", 0.0),
            "superseded_at": sup_days,
            "lifetime_observed": lifetime,
            "is_censored": is_censored,
            "n_reinforcements": len(edge.get("reinforcement_times", [])),
            # Wikipedia-specific: article as entity, no cluster/cohort (to be discovered)
            "entity": edge["subject"],
            "cluster": "unknown",  # to be assigned by clustering
            "cohort": "unknown",
        })

    df = pd.DataFrame(rows)
    return df


def summarize_wikipedia_kg(df: pd.DataFrame) -> dict:
    """Summarize the Wikipedia temporal KG."""
    if df.empty:
        return {"total_edges": 0}

    return {
        "total_edges": len(df),
        "articles": df["subject"].nunique(),
        "predicates": df["predicate"].nunique(),
        "superseded_edges": int((~df["is_censored"]).sum()),
        "censored_edges": int(df["is_censored"].sum()),
        "supersession_rate": float((~df["is_censored"]).mean()),
        "mean_lifetime_days": float(df["lifetime_observed"].mean()),
        "median_lifetime_days": float(df["lifetime_observed"].median()),
        "mean_velocity": float(df["velocity"].mean()),
        "mean_volatility": float(df["volatility"].mean()),
        "predicate_types": {
            "infobox": int(df["predicate"].str.startswith("infobox:").sum()),
            "section": int(df["predicate"].str.startswith("section:").sum()),
            "category": int(df["predicate"].str.startswith("category:").sum()),
            "lead": int(df["predicate"].str.startswith("lead").sum()),
        },
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Fetch a small set for testing
    test_articles = ["BRAF (gene)", "Melanoma", "Bitcoin"]

    print("Fetching article data...")
    article_data = fetch_article_data(
        articles=test_articles,
        max_revisions_per_article=50,
        sample_for_content=20,
    )

    print("\nBuilding temporal KG...")
    df = build_wikipedia_temporal_kg(article_data)

    if not df.empty:
        summary = summarize_wikipedia_kg(df)
        print(f"\nWikipedia Temporal KG:")
        print(f"  Total edges: {summary['total_edges']}")
        print(f"  Articles: {summary['articles']}")
        print(f"  Predicates: {summary['predicates']}")
        print(f"  Superseded: {summary['superseded_edges']} ({summary['supersession_rate']:.1%})")
        print(f"  Mean lifetime: {summary['mean_lifetime_days']:.0f} days")
        print(f"  Predicate types: {summary['predicate_types']}")

        # Show some examples
        print("\nSample edges:")
        for _, row in df.head(10).iterrows():
            status = "superseded" if not row["is_censored"] else "active"
            print(f"  [{row['subject']}] {row['predicate']} = {str(row['object_value'])[:60]}... "
                  f"({status}, lifetime={row['lifetime_observed']:.0f}d)")
    else:
        print("No edges extracted. Check article data.")
