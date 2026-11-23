"""Model assumptions.

Every number that is not measured by the satellite or the weather data lives here,
so it can be checked, challenged and changed in one place. The values are typical
literature ranges for Danish / North-European grazing systems and are meant for a
proof of concept, not for farm-level decisions.
"""
from dataclasses import dataclass


@dataclass
class Assumptions:
    # --- Season -----------------------------------------------------------------
    season_start: str = "04-01"        # grass growing season (MM-DD)
    season_end: str = "10-31"
    grazing_start: str = "05-01"       # main grazing season used for health and cow numbers
    grazing_end: str = "09-30"

    # --- Grass growth (light-use-efficiency model) -------------------------------
    rue_g_dm_per_mj_par: float = 1.0   # radiation use efficiency of temperate grass, typical range 1.0-1.5 (conservative)
    par_fraction: float = 0.48         # share of global radiation that is photosynthetically active
    fapar_slope: float = 1.24          # fAPAR = slope * NDVI + intercept (linear NDVI-fAPAR relation)
    fapar_intercept: float = -0.168
    t_base_c: float = 5.0              # no grass growth below this daily mean temperature
    t_opt_c: float = 10.0              # full growth potential above this temperature
    utilisation: float = 0.75          # share of grown dry matter actually eaten / harvested

    # --- Drought stress (simple soil-water balance) --------------------------------
    soil_water_capacity_mm: float = 80.0   # plant-available water in the root zone (sandy Jutland soils are low)
    stress_threshold: float = 0.5          # growth starts to fall when soil water drops below 50% of capacity
    hargreaves_coef: float = 0.0135        # reference evapotranspiration, Hargreaves radiation formula

    # --- Field health classes (mean NDVI in the grazing season) -------------------
    ndvi_good: float = 0.75
    ndvi_fair: float = 0.60

    # --- Feed quality: NDVI used as a proxy for leafy, energy-rich grass ----------
    ndvi_low_quality: float = 0.55     # at or below this NDVI -> nel_low
    ndvi_high_quality: float = 0.85    # at or above this NDVI -> nel_high
    nel_low: float = 5.8               # MJ NEL per kg DM (stemmy / stressed grass)
    nel_high: float = 6.8              # MJ NEL per kg DM (young leafy grass)

    # --- Cow ---------------------------------------------------------------------
    cow_dmi_kg: float = 20.0           # dry-matter intake per cow per day (Holstein, ~30 kg milk)
    pasture_share: float = 0.5         # share of the daily ration that comes from this pasture
    diet_nel: float = 6.6              # MJ NEL per kg DM of the whole ration
    maintenance_mj_nel: float = 40.0   # energy a ~600 kg cow needs per day before producing milk
    mj_nel_per_kg_ecm: float = 3.14    # energy needed per kg energy-corrected milk
    kg_per_litre: float = 1.03         # milk density

    # --- Supply chain --------------------------------------------------------------
    replacement_feed_nel: float = 6.4      # MJ NEL per kg DM of bought-in feed (silage / concentrate mix)
    replacement_feed_eur_per_kg_dm: float = 0.30
    tanker_capacity_litres: float = 30000.0
    low_growth_alert: float = 0.80         # flag months below 80% of the baseline year
    alert_window_days: int = 30            # early warning: rolling 30-day grass growth vs baseline

    # --- Dairy production (standard conversion, litres of milk per kg of product) ----
    litres_per_kg_cheese: float = 10.0     # hard / semi-hard cheese
    litres_per_kg_butter: float = 22.0     # butter (skimmed milk left over for other products)

    @property
    def maintenance_share(self) -> float:
        """Share of the energy eaten that goes to maintenance instead of milk."""
        return self.maintenance_mj_nel / (self.cow_dmi_kg * self.diet_nel)
