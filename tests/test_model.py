"""Offline tests of the calculation layer with synthetic inputs (no Earth Engine needed)."""
import numpy as np
import pandas as pd
import pytest

from pasture_supply_chain.assumptions import Assumptions
from pasture_supply_chain.model import (analyse_field_year, daily_ndvi, feed_energy,
                                        health_class, supply_chain_impact)

A = Assumptions()


def synthetic_inputs(year, peak_ndvi, summer_dip=0.0):
    days = pd.date_range(f"{year}-04-01", f"{year}-10-31", freq="D")
    doy = days.dayofyear.values
    season = np.sin(np.pi * (doy - 91) / (304 - 91))
    ndvi = 0.35 + (peak_ndvi - 0.35) * season
    ndvi = ndvi - summer_dip * np.exp(-((doy - 200) / 20) ** 2)    # drought in July
    obs = pd.DataFrame({"date": days[::10], "ndvi": ndvi[::10]})  # one clear image every 10 days
    ssrd = 6e6 + 14e6 * np.sin(np.pi * (doy - 60) / 250).clip(0)   # J/m2/day
    temp = 4 + 13 * np.sin(np.pi * (doy - 80) / 250).clip(0)
    wx = pd.DataFrame({"date": days, "ssrd_j_m2": ssrd, "t_mean_c": temp})
    return obs, wx


def test_daily_ndvi_fills_every_day():
    obs = pd.DataFrame({"date": ["2023-04-05", "2023-06-01", "2023-10-20"], "ndvi": [0.4, 0.8, 0.5]})
    s = daily_ndvi(obs, "2023-04-01", "2023-10-31")
    assert len(s) == 214 and s.notna().all()


def test_classes_and_energy():
    assert health_class(0.8, A) == "Good"
    assert health_class(0.65, A) == "Fair"
    assert health_class(0.4, A) == "Poor"
    assert feed_energy(0.9, A) == pytest.approx(A.nel_high)
    assert feed_energy(0.3, A) == pytest.approx(A.nel_low)


def test_healthy_field_gives_realistic_danish_grass_yield():
    obs, wx = synthetic_inputs(2023, peak_ndvi=0.9)
    r = analyse_field_year("test", 2023, 5.0, obs, wx, A)
    # Danish grass / grass-clover typically yields roughly 8-12 t DM/ha per year
    assert 7 < r.dm_yield_t_ha < 14
    assert r.health == "Good"
    assert r.milk_litres > 0 and r.cows_supported > 0


def test_drought_reduces_grass_milk_and_adds_feed_cost():
    obs_b, wx_b = synthetic_inputs(2019, peak_ndvi=0.85)
    obs_t, wx_t = synthetic_inputs(2018, peak_ndvi=0.85, summer_dip=0.35)
    base = [analyse_field_year("f", 2019, 5.0, obs_b, wx_b, A)]
    drought = [analyse_field_year("f", 2018, 5.0, obs_t, wx_t, A)]
    summary, monthly = supply_chain_impact(drought, base, A)
    assert summary["grass_change_pct"] < 0
    assert summary["milk_change_litres"] < 0
    assert summary["extra_feed_cost_eur"] > 0
    assert "Jul" in summary["alert_months"]
    assert len(monthly) == 7
