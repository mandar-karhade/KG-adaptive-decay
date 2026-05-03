"""Synthea clinical data pipeline for temporal KG construction.

Loads Synthea CSV exports and builds a temporal KG where:
- Observations (vitals, labs) are edges with numeric/text values that get superseded
- Conditions (diagnoses) are edges that persist or resolve
- Medications are edges that start and stop (superseded when treatment changes)

Each edge gets velocity and volatility computed from the patient's history.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/synthea/csv")


def load_observations(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and parse Synthea observations (vitals, labs)."""
    if data_dir is None:
        data_dir = DATA_DIR
    df = pd.read_csv(data_dir / "observations.csv", low_memory=False)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values(["PATIENT", "CODE", "DATE"])
    return df


def load_conditions(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and parse Synthea conditions (diagnoses)."""
    if data_dir is None:
        data_dir = DATA_DIR
    df = pd.read_csv(data_dir / "conditions.csv", low_memory=False)
    df["START"] = pd.to_datetime(df["START"])
    df["STOP"] = pd.to_datetime(df["STOP"], errors="coerce")
    return df


def load_medications(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and parse Synthea medications."""
    if data_dir is None:
        data_dir = DATA_DIR
    df = pd.read_csv(data_dir / "medications.csv", low_memory=False)
    df["START"] = pd.to_datetime(df["START"])
    df["STOP"] = pd.to_datetime(df["STOP"], errors="coerce")
    return df


def load_encounters(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and parse Synthea encounters."""
    if data_dir is None:
        data_dir = DATA_DIR
    df = pd.read_csv(data_dir / "encounters.csv", low_memory=False)
    df["START"] = pd.to_datetime(df["START"])
    df["STOP"] = pd.to_datetime(df["STOP"], errors="coerce")
    return df


def build_observation_edges(obs_df: pd.DataFrame, enc_df: pd.DataFrame) -> list[dict]:
    """Build temporal KG edges from observations.

    For each (patient, observation_type) concept, track value changes over time.
    A new value that differs from the previous is a supersession.
    A re-observation with the same value is a reinforcement.
    """
    # Map encounter to encounter class (ICU proxy: inpatient vs outpatient vs emergency)
    enc_class = enc_df.set_index("Id")["ENCOUNTERCLASS"].to_dict()

    edges = []
    edge_id = 0

    # Group by patient + observation code (= concept)
    for (patient, code), group in obs_df.groupby(["PATIENT", "CODE"]):
        group = group.sort_values("DATE")
        description = group["DESCRIPTION"].iloc[0]

        current_value = None
        current_edge = None

        for _, row in group.iterrows():
            value = str(row["VALUE"]) if pd.notna(row["VALUE"]) else ""
            timestamp = row["DATE"]
            encounter_id = row["ENCOUNTER"]
            care_setting = enc_class.get(encounter_id, "unknown")

            if current_value is None:
                # First observation
                current_value = value
                current_edge = {
                    "edge_id": edge_id,
                    "subject": patient[:8],  # truncate UUID for readability
                    "predicate": f"obs:{code}",
                    "predicate_description": description,
                    "object_value": value,
                    "timestamp": timestamp,
                    "care_setting": care_setting,
                    "category": "observation",
                    "reinforcement_times": [],
                    "superseded_at": None,
                }
                edge_id += 1
            elif value == current_value:
                # Reinforcement (same value)
                current_edge["reinforcement_times"].append(timestamp)
            else:
                # Supersession (different value)
                current_edge["superseded_at"] = timestamp
                edges.append(current_edge)

                current_value = value
                current_edge = {
                    "edge_id": edge_id,
                    "subject": patient[:8],
                    "predicate": f"obs:{code}",
                    "predicate_description": description,
                    "object_value": value,
                    "timestamp": timestamp,
                    "care_setting": care_setting,
                    "category": "observation",
                    "reinforcement_times": [],
                    "superseded_at": None,
                }
                edge_id += 1

        # Last edge is right-censored
        if current_edge is not None:
            edges.append(current_edge)

    return edges


def build_condition_edges(cond_df: pd.DataFrame) -> list[dict]:
    """Build temporal KG edges from conditions.

    Conditions with a STOP date are superseded (resolved/changed).
    Conditions without a STOP date are right-censored (still active).
    """
    edges = []
    edge_id_start = 1_000_000  # offset to avoid collision with observations

    for i, row in cond_df.iterrows():
        edge = {
            "edge_id": edge_id_start + i,
            "subject": row["PATIENT"][:8],
            "predicate": f"cond:{row['CODE']}",
            "predicate_description": row["DESCRIPTION"],
            "object_value": row["DESCRIPTION"],
            "timestamp": row["START"],
            "care_setting": "diagnosis",
            "category": "condition",
            "reinforcement_times": [],
            "superseded_at": row["STOP"] if pd.notna(row["STOP"]) else None,
        }
        edges.append(edge)

    return edges


def build_medication_edges(med_df: pd.DataFrame) -> list[dict]:
    """Build temporal KG edges from medications.

    Medications with a STOP date are superseded (discontinued/changed).
    Active medications are right-censored.
    """
    edges = []
    edge_id_start = 2_000_000

    for i, row in med_df.iterrows():
        edge = {
            "edge_id": edge_id_start + i,
            "subject": row["PATIENT"][:8],
            "predicate": f"med:{row['CODE']}",
            "predicate_description": row["DESCRIPTION"],
            "object_value": row["DESCRIPTION"],
            "timestamp": row["START"],
            "care_setting": "medication",
            "category": "medication",
            "reinforcement_times": [],
            "superseded_at": row["STOP"] if pd.notna(row["STOP"]) else None,
        }
        edges.append(edge)

    return edges


def edges_to_dataframe(edges: list[dict], reference_date: datetime | None = None) -> pd.DataFrame:
    """Convert edge list to DataFrame matching the format expected by the framework."""
    if reference_date is None:
        # Use the latest timestamp as reference for right-censoring
        all_timestamps = [pd.Timestamp(e["timestamp"]).tz_localize(None) for e in edges]
        reference_date = max(all_timestamps)

    rows = []
    for edge in edges:
        ts = pd.Timestamp(edge["timestamp"]).tz_localize(None)
        sup_at = edge["superseded_at"]

        if sup_at is not None and pd.notna(sup_at):
            sup_at = pd.Timestamp(sup_at).tz_localize(None)
            lifetime = (sup_at - ts).total_seconds() / 86400  # days
            is_censored = False
        else:
            lifetime = (reference_date - ts).total_seconds() / 86400
            is_censored = True

        if lifetime <= 0:
            continue

        rows.append({
            "edge_id": edge["edge_id"],
            "subject": edge["subject"],
            "predicate": edge["predicate"],
            "predicate_description": edge["predicate_description"],
            "object_value": str(edge["object_value"])[:200],
            "timestamp": (ts - pd.Timestamp("2000-01-01")).total_seconds() / 86400,
            "care_setting": edge["care_setting"],
            "category": edge["category"],
            "lifetime_observed": lifetime,
            "is_censored": is_censored,
            "n_reinforcements": len(edge["reinforcement_times"]),
            # Placeholder: velocity and volatility computed after aggregation
            "velocity": 0.0,
            "volatility": 0.0,
            # For compatibility with existing framework
            "entity": edge["subject"],
            "cluster": "unknown",
            "cohort": edge["care_setting"],
        })

    return pd.DataFrame(rows)


def compute_concept_velocity_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Compute velocity and volatility per concept (patient + predicate)."""
    df = df.copy()

    # Velocity: observations per day per concept
    concept_counts = df.groupby(["subject", "predicate"]).agg(
        n_edges=("edge_id", "count"),
        timespan=("lifetime_observed", "sum"),
    )
    concept_counts["velocity"] = concept_counts["n_edges"] / concept_counts["timespan"].clip(lower=1.0)

    # Volatility: fraction of edges that were superseded (value changed)
    concept_vol = df.groupby(["subject", "predicate"]).agg(
        n_superseded=("is_censored", lambda x: (~x).sum()),
        n_total=("edge_id", "count"),
    )
    concept_vol["volatility"] = concept_vol["n_superseded"] / concept_vol["n_total"].clip(lower=1)

    # Merge back
    vel_map = concept_counts["velocity"].to_dict()
    vol_map = concept_vol["volatility"].to_dict()

    df["velocity"] = df.apply(lambda r: vel_map.get((r["subject"], r["predicate"]), 0.0), axis=1)
    df["volatility"] = df.apply(lambda r: vol_map.get((r["subject"], r["predicate"]), 0.0), axis=1)

    return df


def build_synthea_temporal_kg(data_dir: Path | None = None) -> pd.DataFrame:
    """Full pipeline: load Synthea data, build temporal KG."""
    logger.info("Loading Synthea data...")
    obs_df = load_observations(data_dir)
    cond_df = load_conditions(data_dir)
    med_df = load_medications(data_dir)
    enc_df = load_encounters(data_dir)

    logger.info(f"  Observations: {len(obs_df)}, Conditions: {len(cond_df)}, "
                f"Medications: {len(med_df)}, Encounters: {len(enc_df)}")

    logger.info("Building edges...")
    obs_edges = build_observation_edges(obs_df, enc_df)
    cond_edges = build_condition_edges(cond_df)
    med_edges = build_medication_edges(med_df)

    all_edges = obs_edges + cond_edges + med_edges
    logger.info(f"  Total edges: {len(all_edges)} "
                f"(obs={len(obs_edges)}, cond={len(cond_edges)}, med={len(med_edges)})")

    logger.info("Converting to DataFrame...")
    df = edges_to_dataframe(all_edges)

    logger.info("Computing velocity and volatility...")
    df = compute_concept_velocity_volatility(df)

    return df


def summarize_synthea_kg(df: pd.DataFrame) -> dict:
    """Summarize the Synthea temporal KG."""
    return {
        "total_edges": len(df),
        "patients": df["subject"].nunique(),
        "predicates": df["predicate"].nunique(),
        "categories": df["category"].value_counts().to_dict(),
        "care_settings": df["care_setting"].value_counts().to_dict(),
        "supersession_rate": float((~df["is_censored"]).mean()),
        "mean_lifetime_days": float(df["lifetime_observed"].mean()),
        "median_lifetime_days": float(df["lifetime_observed"].median()),
        "by_category": {
            cat: {
                "n": int(sub["edge_id"].count()),
                "supersession_rate": float((~sub["is_censored"]).mean()),
                "median_lifetime": float(sub["lifetime_observed"].median()),
                "mean_velocity": float(sub["velocity"].mean()),
                "mean_volatility": float(sub["volatility"].mean()),
            }
            for cat, sub in df.groupby("category")
        },
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    print("Building Synthea temporal KG...")
    df = build_synthea_temporal_kg()

    summary = summarize_synthea_kg(df)
    print(f"\nSynthea Temporal KG:")
    print(f"  Total edges: {summary['total_edges']}")
    print(f"  Patients: {summary['patients']}")
    print(f"  Predicates: {summary['predicates']}")
    print(f"  Supersession rate: {summary['supersession_rate']:.1%}")
    print(f"  Median lifetime: {summary['median_lifetime_days']:.0f} days")

    print("\nBy category:")
    for cat, stats in summary["by_category"].items():
        print(f"  {cat}: n={stats['n']}, sup_rate={stats['supersession_rate']:.2f}, "
              f"med_life={stats['median_lifetime']:.0f}d, "
              f"vel={stats['mean_velocity']:.4f}, vol={stats['mean_volatility']:.3f}")

    # Save
    df.to_parquet("data/synthea/temporal_kg.parquet", index=False)
    print(f"\nSaved to data/synthea/temporal_kg.parquet")
