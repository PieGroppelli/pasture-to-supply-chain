"""Offline tests of the calculation layer with synthetic inputs (no Earth Engine needed)."""
import numpy as np
import pandas as pd
import pytest

from pasture_supply_chain.assumptions import Assumptions
from pasture_supply_chain.model import (analyse_field_year, daily_ndvi, feed_energy,
                                        health_class, supply_chain_impact, water_balance)

A = Assumptions()


def synthetic_inputs(year, peak_ndvi, summer_dip=0.0, rain_mm_day=2.2):
    days = pd.date_range(f"{year}-04-01", f"{year}-10-31", freq="D")
    doy = days.dayofyear.values
    season = np.sin(np.pi * (doy - 91) / (304 - 91))
    ndvi = 0.35 + (peak_ndvi - 0.35) * season
    ndvi = ndvi - summer_dip * np.exp(-((doy - 200) / 20) ** 2)    # drought in July
    obs = pd.DataFrame({"date": days[::10], "ndvi": ndvi[::10]})  # one clear image every 10 days
    ssrd = 6e6 + 14e6 * np.sin(np.pi * (doy - 60) / 250).clip(0)   # J/m2/day
    temp = 4 + 13 * np.sin(np.pi * (doy - 80) / 250).clip(0)
    rain = np.full(len(days), rain_mm_day)
    wx = pd.DataFrame({"date": days, "ssrd_j_m2": ssrd, "t_mean_c": temp, "precip_mm": rain})
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
    # Danish grass / grass-clover typically yields roughly 6-12 t DM/ha per year
    assert 6 < r.dm_yield_t_ha < 14
    assert r.health == "Good"
    assert r.milk_litres > 0 and r.cows_supported > 0


def test_drought_reduces_grass_milk_and_adds_feed_cost():
    obs_b, wx_b = synthetic_inputs(2019, peak_ndvi=0.85)
    obs_t, wx_t = synthetic_inputs(2018, peak_ndvi=0.85, summer_dip=0.35, rain_mm_day=0.8)
    base = [analyse_field_year("f", 2019, 5.0, obs_b, wx_b, A)]
    drought = [analyse_field_year("f", 2018, 5.0, obs_t, wx_t, A)]
    summary, monthly = supply_chain_impact(drought, base, A)
    assert summary["grass_change_pct"] < 0
    assert summary["milk_change_litres"] < 0
    assert summary["extra_feed_cost_eur"] > 0
    assert "Jul" in summary["alert_months"]
    assert len(monthly) == 7
    # early warning comes before the end of the worst month, and missing milk is translated into product
    assert summary["first_alert_date"] is not None
    assert summary["warning_days_before_end_of_worst_month"] > 0
    assert summary["missing_milk_as_cheese_t"] > summary["missing_milk_as_butter_t"] > 0


def test_no_rain_creates_water_stress():
    n = 120
    f_dry, sw_dry = water_balance(np.full(n, 18e6), np.full(n, 16.0), np.zeros(n), A)
    f_wet, _ = water_balance(np.full(n, 18e6), np.full(n, 16.0), np.full(n, 5.0), A)
    assert f_dry[0] == 1.0 and f_dry[-1] < 0.2 and sw_dry[-1] < 10
    assert f_wet.min() == 1.0


def test_dry_sunny_year_grows_less_than_wet_year_even_with_same_ndvi():
    # The first real run showed that sunshine alone made 2018 look better than 2019:
    # with the water balance, a sunny but dry season must grow less grass.
    obs, wx = synthetic_inputs(2018, peak_ndvi=0.85, rain_mm_day=2.2)
    wet = analyse_field_year("f", 2018, 5.0, obs, wx, A)
    dry_wx = wx.assign(precip_mm=0.6, ssrd_j_m2=wx["ssrd_j_m2"] * 1.15)
    dry = analyse_field_year("f", 2018, 5.0, obs, dry_wx, A)
    assert dry.dm_yield_t_ha < wet.dm_yield_t_ha
    assert dry.drought_days > wet.drought_days
