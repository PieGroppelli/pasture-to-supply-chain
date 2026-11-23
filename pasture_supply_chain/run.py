"""End-to-end pipeline: fields -> health -> grass -> cows -> milk -> supply chain."""
from __future__ import annotations

import json
import os

import pandas as pd

from . import report, satellite
from .assumptions import Assumptions
from .model import analyse_field_year, supply_chain_impact


def run(project_id: str, fields_path: str, target_year: int, baseline_year: int,
        out_dir: str = "outputs", assumptions: Assumptions | None = None, verbose: bool = True):
    a = assumptions or Assumptions()
    os.makedirs(out_dir, exist_ok=True)
    satellite.init(project_id)
    fields = satellite.load_fields(fields_path)

    results = {target_year: [], baseline_year: []}
    images, similarity = {}, {}

    for f in fields:
        for year in (target_year, baseline_year):
            start, end = f"{year}-{a.season_start}", f"{year}-{a.season_end}"
            obs = satellite.ndvi_observations(f["geometry"], start, end)
            if obs.empty:
                raise RuntimeError(f"No cloud-free Sentinel-2 images for {f['name']} in {year}")
            wx = satellite.weather(f["geometry"], start, end)
            results[year].append(analyse_field_year(f["name"], year, f["area_ha"], obs, wx, a))
        images[f["name"]] = (satellite.thumbnail(f["geometry"], target_year, "rgb"),
                             satellite.thumbnail(f["geometry"], target_year, "ndvi"))
        similarity[f["name"]] = satellite.embedding_similarity(f["geometry"], baseline_year, target_year)
        if verbose:
            print(f"done: {f['name']}")

    # Tables
    table = pd.DataFrame([r.summary() for y in (target_year, baseline_year) for r in results[y]])
    table["embedding_similarity_vs_other_year"] = table["field"].map(similarity)
    table.to_csv(os.path.join(out_dir, "field_results.csv"), index=False)

    impact, monthly = supply_chain_impact(results[target_year], results[baseline_year], a)
    monthly.to_csv(os.path.join(out_dir, "monthly_supply_chain.csv"))
    with open(os.path.join(out_dir, "supply_chain_summary.json"), "w") as fh:
        json.dump({"target_year": target_year, "baseline_year": baseline_year, **impact}, fh, indent=2)

    # Charts
    report.field_maps(fields, images, results[target_year], os.path.join(out_dir, "field_maps.png"))
    report.ndvi_curves(results[target_year], results[baseline_year], os.path.join(out_dir, "ndvi_curves.png"))
    report.supply_chain_chart(monthly, target_year, baseline_year, os.path.join(out_dir, "supply_chain.png"))

    if verbose:
        print("\nFIELD RESULTS")
        print(table.to_string(index=False))
        print("\nSUPPLY CHAIN IMPACT", target_year, "vs", baseline_year)
        for k, v in impact.items():
            print(f"  {k}: {v}")
    return table, impact, monthly
