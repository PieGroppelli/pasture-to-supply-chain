"""Pure calculation layer: from NDVI and weather to grass, cows, milk and supply chain.

No Earth Engine calls here, so the logic can be tested offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .assumptions import Assumptions


# ---------------------------------------------------------------------------
# 1. Time series preparation
# ---------------------------------------------------------------------------
def daily_ndvi(obs: pd.DataFrame, start: str, end: str, window: int = 15) -> pd.Series:
    """Turn irregular cloud-free NDVI observations into a smooth daily series.

    obs: DataFrame with columns 'date' and 'ndvi'.
    """
    s = (obs.assign(date=pd.to_datetime(obs["date"]))
            .groupby("date")["ndvi"].mean().sort_index())
    idx = pd.date_range(start, end, freq="D")
    s = s.reindex(s.index.union(idx)).interpolate(method="time").reindex(idx)
    s = s.bfill().ffill()
    return s.rolling(window, center=True, min_periods=1).median().rename("ndvi")


# ---------------------------------------------------------------------------
# 2. Grass growth
# ---------------------------------------------------------------------------
def fapar_from_ndvi(ndvi, a: Assumptions):
    return np.clip(a.fapar_slope * np.asarray(ndvi) + a.fapar_intercept, 0.0, 0.95)


def temperature_factor(t_mean_c, a: Assumptions):
    return np.clip((np.asarray(t_mean_c) - a.t_base_c) / (a.t_opt_c - a.t_base_c), 0.0, 1.0)


def daily_growth_kg_dm_ha(ndvi, ssrd_j_m2, t_mean_c, a: Assumptions):
    """Daily grass growth in kg dry matter per hectare.

    growth = RUE x fAPAR x PAR x temperature factor
    1 g/m2 = 10 kg/ha
    """
    par_mj_m2 = np.asarray(ssrd_j_m2) / 1e6 * a.par_fraction
    g_m2 = a.rue_g_dm_per_mj_par * fapar_from_ndvi(ndvi, a) * par_mj_m2 * temperature_factor(t_mean_c, a)
    return g_m2 * 10.0


# ---------------------------------------------------------------------------
# 3. Field health and feed quality
# ---------------------------------------------------------------------------
def health_class(mean_ndvi: float, a: Assumptions) -> str:
    if mean_ndvi >= a.ndvi_good:
        return "Good"
    if mean_ndvi >= a.ndvi_fair:
        return "Fair"
    return "Poor"


def feed_energy(mean_ndvi: float, a: Assumptions) -> float:
    """Estimated energy density of the grass (MJ NEL / kg DM), linear in NDVI."""
    x = (mean_ndvi - a.ndvi_low_quality) / (a.ndvi_high_quality - a.ndvi_low_quality)
    return float(a.nel_low + np.clip(x, 0.0, 1.0) * (a.nel_high - a.nel_low))


# ---------------------------------------------------------------------------
# 4. Cows and milk
# ---------------------------------------------------------------------------
def milk_from_dm(utilisable_dm_kg, nel: float, a: Assumptions):
    """Energy-corrected milk (kg) that the eaten grass can support."""
    return np.asarray(utilisable_dm_kg) * nel * (1 - a.maintenance_share) / a.mj_nel_per_kg_ecm


def grazing_days(a: Assumptions, year: int) -> int:
    return len(pd.date_range(f"{year}-{a.grazing_start}", f"{year}-{a.grazing_end}", freq="D"))


# ---------------------------------------------------------------------------
# 5. One field, one year
# ---------------------------------------------------------------------------
@dataclass
class FieldYear:
    field: str
    year: int
    area_ha: float
    n_obs: int
    mean_ndvi: float
    peak_ndvi: float
    health: str
    dm_yield_t_ha: float
    utilisable_dm_t: float
    feed_nel: float
    cows_supported: float
    milk_ecm_kg: float
    milk_litres: float
    daily: pd.DataFrame = field(repr=False, default=None)
    monthly: pd.DataFrame = field(repr=False, default=None)

    def summary(self) -> dict:
        return {
            "field": self.field, "year": self.year, "area_ha": round(self.area_ha, 2),
            "cloud_free_images": self.n_obs, "mean_ndvi": round(self.mean_ndvi, 3),
            "peak_ndvi": round(self.peak_ndvi, 3), "health": self.health,
            "grass_yield_t_dm_ha": round(self.dm_yield_t_ha, 2),
            "utilisable_dm_t": round(self.utilisable_dm_t, 2),
            "feed_energy_mj_nel_kg_dm": round(self.feed_nel, 2),
            "cows_supported_grazing_season": round(self.cows_supported, 1),
            "milk_from_pasture_kg_ecm": round(self.milk_ecm_kg),
            "milk_from_pasture_litres": round(self.milk_litres),
        }


def analyse_field_year(name: str, year: int, area_ha: float, ndvi_obs: pd.DataFrame,
                       weather: pd.DataFrame, a: Assumptions) -> FieldYear:
    """weather: DataFrame with columns 'date', 'ssrd_j_m2', 't_mean_c'."""
    start, end = f"{year}-{a.season_start}", f"{year}-{a.season_end}"
    ndvi = daily_ndvi(ndvi_obs, start, end)

    w = weather.assign(date=pd.to_datetime(weather["date"])).set_index("date").sort_index()
    w = w.reindex(ndvi.index).interpolate(method="time").bfill().ffill()

    growth = daily_growth_kg_dm_ha(ndvi.values, w["ssrd_j_m2"].values, w["t_mean_c"].values, a)
    daily = pd.DataFrame({"ndvi": ndvi.values, "growth_kg_dm_ha": growth}, index=ndvi.index)

    graze = daily.loc[f"{year}-{a.grazing_start}":f"{year}-{a.grazing_end}"]
    mean_ndvi = float(graze["ndvi"].mean())
    nel = feed_energy(mean_ndvi, a)

    dm_yield_t_ha = float(daily["growth_kg_dm_ha"].sum()) / 1000.0
    utilisable_dm_kg = dm_yield_t_ha * 1000.0 * area_ha * a.utilisation

    cows = utilisable_dm_kg / (a.cow_dmi_kg * a.pasture_share * grazing_days(a, year))
    milk_ecm = float(milk_from_dm(utilisable_dm_kg, nel, a))

    monthly = daily["growth_kg_dm_ha"].resample("MS").sum().to_frame("growth_kg_dm_ha")
    monthly["utilisable_dm_kg"] = monthly["growth_kg_dm_ha"] * area_ha * a.utilisation
    monthly["milk_litres"] = milk_from_dm(monthly["utilisable_dm_kg"], nel, a) / a.kg_per_litre

    return FieldYear(
        field=name, year=year, area_ha=area_ha, n_obs=int(len(ndvi_obs)),
        mean_ndvi=mean_ndvi, peak_ndvi=float(daily["ndvi"].max()),
        health=health_class(mean_ndvi, a), dm_yield_t_ha=dm_yield_t_ha,
        utilisable_dm_t=utilisable_dm_kg / 1000.0, feed_nel=nel, cows_supported=cows,
        milk_ecm_kg=milk_ecm, milk_litres=milk_ecm / a.kg_per_litre,
        daily=daily, monthly=monthly,
    )


# ---------------------------------------------------------------------------
# 6. Supply chain impact: target year vs baseline year, all fields together
# ---------------------------------------------------------------------------
def supply_chain_impact(target: list[FieldYear], baseline: list[FieldYear], a: Assumptions):
    """Returns (summary dict, monthly DataFrame)."""
    def monthly_total(results):
        df = sum((r.monthly.reset_index(drop=True) for r in results))
        df.index = [m.strftime("%b") for m in results[0].monthly.index]
        return df

    mt, mb = monthly_total(target), monthly_total(baseline)
    monthly = pd.DataFrame({
        "milk_litres_target": mt["milk_litres"],
        "milk_litres_baseline": mb["milk_litres"],
    })
    monthly["growth_vs_baseline"] = mt["utilisable_dm_kg"] / mb["utilisable_dm_kg"].replace(0, np.nan)
    monthly["tanker_loads_target"] = monthly["milk_litres_target"] / a.tanker_capacity_litres
    monthly["tanker_loads_baseline"] = monthly["milk_litres_baseline"] / a.tanker_capacity_litres
    monthly["alert"] = monthly["growth_vs_baseline"] < a.low_growth_alert

    dm_t = sum(r.utilisable_dm_t for r in target)
    dm_b = sum(r.utilisable_dm_t for r in baseline)
    milk_t = sum(r.milk_litres for r in target)
    milk_b = sum(r.milk_litres for r in baseline)

    # Energy the herd misses from grass -> bought-in feed needed to keep milk output stable
    energy_t = sum(r.utilisable_dm_t * 1000 * r.feed_nel for r in target)
    energy_b = sum(r.utilisable_dm_t * 1000 * r.feed_nel for r in baseline)
    missing_mj = max(energy_b - energy_t, 0.0)
    extra_feed_kg = missing_mj / a.replacement_feed_nel
    extra_cost = extra_feed_kg * a.replacement_feed_eur_per_kg_dm

    area = sum(r.area_ha for r in target)
    per_1000_ha = 1000 / area if area else 0.0

    summary = {
        "total_area_ha": round(area, 1),
        "utilisable_grass_t_dm_target": round(dm_t, 1),
        "utilisable_grass_t_dm_baseline": round(dm_b, 1),
        "grass_change_pct": round((dm_t / dm_b - 1) * 100, 1) if dm_b else None,
        "milk_from_pasture_litres_target": round(milk_t),
        "milk_from_pasture_litres_baseline": round(milk_b),
        "milk_change_litres": round(milk_t - milk_b),
        "extra_feed_needed_t_dm": round(extra_feed_kg / 1000, 1),
        "extra_feed_cost_eur": round(extra_cost),
        "extra_cost_eur_per_1000_litres": round(extra_cost / milk_b * 1000, 1) if milk_b else None,
        "tanker_loads_change": round((milk_t - milk_b) / a.tanker_capacity_litres, 1),
        "alert_months": [m for m, flag in monthly["alert"].items() if flag],
        # Same result scaled to a 1,000 ha milk-collection area, to read it at supply-chain level
        "scaled_milk_change_litres_per_1000_ha": round((milk_t - milk_b) * per_1000_ha),
        "scaled_extra_feed_cost_eur_per_1000_ha": round(extra_cost * per_1000_ha),
        "scaled_tanker_loads_change_per_1000_ha": round((milk_t - milk_b) * per_1000_ha / a.tanker_capacity_litres, 1),
    }
    return summary, monthly
