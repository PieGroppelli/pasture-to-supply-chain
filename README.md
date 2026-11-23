# From satellite to supply chain: pasture health, milk and logistics impact

A small independent project that follows dairy from the field to the supply chain.
Using free satellite and weather data, it checks how healthy a set of grazing fields is,
estimates how much grass they grow, how many cows that grass can feed and how much milk
it supports, and then translates a bad season into what it means for the supply chain:
less milk to collect, less cheese or butter the dairies can make, extra feed to buy, and how
many weeks of warning planners get before the worst month.

The test case compares **2018, the drought year in Denmark, with 2019**, on three grassland
parcels in Jutland. The parcels are not drawn by hand: the code picks them automatically where
Google Dynamic World sees grass in both years and ESA WorldCover maps grassland.

> Proof of concept. The satellite and weather data are real; the agronomic and cost
> parameters are typical literature values collected in one file and clearly marked as
> assumptions. The parcels are chosen from public land-cover data and are not linked to any specific farm.

## What it does

| Step | Question | Data / method |
|---|---|---|
| 0. Find pastures | Where are real grazing fields? | Dynamic World "grass" in every year + ESA WorldCover grassland, compact patches of 3-20 ha |
| 1. See the field | What does the field look like? | Sentinel-2 true-colour and NDVI summer composites |
| 2. Field health | Is the grass healthy this season? | Cloud-masked Sentinel-2 NDVI time series → Good / Fair / Poor |
| 3. Land change | Did the field change character between years (ploughed, reseeded, new use)? | Cosine similarity of Google AlphaEarth satellite embeddings |
| 4. Grass growth | How much dry matter does it grow, and when does drought stop it? | Light-use-efficiency model: RUE × fAPAR(NDVI) × PAR × temperature × water stress, with a daily soil-water balance (ERA5-Land rain, Hargreaves evapotranspiration) |
| 5. Cows | How many cows can graze it? | Utilisable dry matter ÷ (daily intake × pasture share × grazing days) |
| 6. Milk | How much milk does that grass support, and how good is the feed? | Feed energy (MJ NEL/kg DM) from NDVI → energy-corrected milk |
| 7. Planning & logistics | How much milk will be missing, and when? | Monthly milk intake vs a normal year, tanker loads, months below the alert threshold |
| 8. Production | What does the missing milk mean for the dairies? | Missing milk expressed as tonnes of cheese or butter (standard litres per kg) |
| 9. Early warning | How early can planners see it coming? | First day the rolling 30-day grass growth falls below 80% of the normal year, and days of warning before the worst month |
| 10. Cost & sustainability | What does it cost the farmers? | Extra bought-in feed (t DM) and cost per 1,000 L |

## Outputs

- `field_maps.png`: true colour and NDVI for each field, with the health class
- `ndvi_curves.png`: NDVI through the season, target year vs baseline year
- `supply_chain.png`: monthly milk supported by the fields (low-growth months in red) and the early-warning signal
- `field_results.csv`, `monthly_supply_chain.csv`, `supply_chain_summary.json`

## Run it

Easiest is Google Colab: open `notebooks/run_in_colab.ipynb`, add your free
[Earth Engine](https://code.earthengine.google.com/register) project ID and run the cells.

Locally:

```bash
pip install -r requirements.txt
python -c "from pasture_supply_chain.run import run; run('your-ee-project', 'fields/example_fields.geojson', 2018, 2019)"
```

Fields: by default the notebook picks one grassland parcel near each location in `AREAS`.
To use your own fields instead, draw polygons on [geojson.io](https://geojson.io), give each one a `name`
property and save the file in `fields/`. Danish field boundaries are also published by
the Danish Agricultural Agency (Markkort).

## Tests

The calculation layer has no Earth Engine dependency and is tested offline with synthetic data:

```bash
pytest
```

## Project structure

```
pasture_supply_chain/
  assumptions.py   every non-measured parameter, in one place
  satellite.py     Earth Engine: Sentinel-2 NDVI, ERA5-Land weather, AlphaEarth embeddings, images
  model.py         grass growth, health, cows, milk and supply-chain impact (pure Python)
  report.py        charts
  run.py           end-to-end pipeline
fields/            example field polygons (GeoJSON)
notebooks/         Colab notebook
tests/             offline tests
```

## Limitations and next steps

- NDVI saturates in very dense grass and is only a proxy for feed quality; real deployment
  would calibrate against grass samples and actual milk recordings.
- Milk composition (fat, protein) is not predicted; feed energy is used as the quality signal.
- Weather comes from a ~9 km grid, so local rainfall differences are smoothed out.
- The soil-water balance uses one soil capacity for all fields and ignores irrigation, which is common on sandy soils in West Jutland.
- Next steps: calibrate on farms with known yields, add soil moisture, run it at the scale
  of a whole milk-collection area, and connect the monthly forecast to collection planning.

## Data sources

- Sentinel-2 Level-2A / Level-1C, Copernicus (ESA) via Google Earth Engine
- ERA5-Land daily aggregates, ECMWF / Copernicus Climate Change Service
- Satellite Embedding V1 (AlphaEarth Foundations), Google

Author: Pietro Groppelli, M.Sc. Operations and Innovation Management, Aalborg University
