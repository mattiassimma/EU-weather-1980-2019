---
license: cc-by-4.0
language:
- en
task_categories:
- tabular-classification
- tabular-regression
- time-series-forecasting
multilinguality: monolingual
size_categories:
- 1K<n<10K
tags:
- tabular
- europe
- our-world-in-data
- average-precipitation-per-year
- owid
- long-run-series
- time-series
pretty_name: "Average Precipitation Per Year | Europe (Our World in Data)"
---

# Average Precipitation Per Year | Europe (Our World in Data)

🇪🇺 **3,354 observations** · **39 Europe countries** · **1940–2025** · *Repackaged by [Electric Sheep Europe](https://huggingface.co/electricsheepeurope)*

![rows](https://img.shields.io/badge/rows-3,354-blue) ![countries](https://img.shields.io/badge/countries-39-green) ![years](https://img.shields.io/badge/years-1940–2025-orange) ![license](https://img.shields.io/badge/license-cc-by-4.0-lightgrey)

## TL;DR

This dataset contains **3,354 observations** of `Average Precipitation Per Year` data across **39 Europe countries**, spanning **1940–2025**.

## About the source

- **Source:** [Our World in Data](https://ourworldindata.org/grapher/average-precipitation-per-year)
- **Publisher:** Our World in Data
- **License:** [cc-by-4.0](https://creativecommons.org/licenses/by/4.0/)
- **Topic:** Average Precipitation Per Year

## Geographic coverage

39 Europe countries · top rows shown below, sorted by row count:

| Country | Rows | First year | Last year |
|---------|-----:|-----------:|----------:|
| `ALB` | 86 | 1940 | 2025 |
| `AND` | 86 | 1940 | 2025 |
| `AUT` | 86 | 1940 | 2025 |
| `BEL` | 86 | 1940 | 2025 |
| `BGR` | 86 | 1940 | 2025 |
| `BIH` | 86 | 1940 | 2025 |
| `BLR` | 86 | 1940 | 2025 |
| `CHE` | 86 | 1940 | 2025 |
| `CZE` | 86 | 1940 | 2025 |
| `DEU` | 86 | 1940 | 2025 |
| `DNK` | 86 | 1940 | 2025 |
| `ESP` | 86 | 1940 | 2025 |
| `EST` | 86 | 1940 | 2025 |
| `FIN` | 86 | 1940 | 2025 |
| `FRA` | 86 | 1940 | 2025 |
| ... | _24 more countries_ | | |

## Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `country_name` | `string` | — | `Albania` |
| `country_iso3` | `string` | — | `ALB` |
| `year` | `int64` | — | `1940` |
| `Annual precipitation` | `float64` | — | `1388.5201` |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("electricsheepeurope/europe-owid-average-precipitation-per-year")
df = ds["train"].to_pandas()
print(df.head())
```

### Filter to one country

```python
germany = df[df["country_iso3"] == "DEU"]
```

### Time-series for a single indicator

```python
sample = df.sort_values("year")
sample.plot(x="year", y="Annual precipitation")
```

## Citation

```bibtex
@misc{europe_owid_average_precipitation_per_year_2025,
  title        = {Average Precipitation Per Year | Europe (Our World in Data)},
  author       = {Our World in Data},
  year         = {2025},
  url          = {https://ourworldindata.org/grapher/average-precipitation-per-year},
  publisher    = {HuggingFace Datasets, repackaged by Electric Sheep Europe},
  howpublished = {\url{https://huggingface.co/datasets/electricsheepeurope/europe-owid-average-precipitation-per-year}}
}
```

## License

Released under [cc-by-4.0](https://creativecommons.org/licenses/by/4.0/).

Original data © Our World in Data. When using this dataset, please cite both the original source above and the Electric Sheep Europe repackaging.

## About Electric Sheep

Electric Sheep Europe is part of the Electric Sheep mission: a unified, ML-ready data layer for Europe on HuggingFace. We pull data from authoritative open sources, normalize the schemas, package as Parquet, and publish with consistent dataset cards so researchers and developers can use `load_dataset()` to start working in seconds.

Browse the full collection: [huggingface.co/electricsheepeurope](https://huggingface.co/electricsheepeurope)

---

_Provenance: ingested 2026-06-02 via the Electric Sheep pipeline. Source URL: https://ourworldindata.org/grapher/average-precipitation-per-year_
