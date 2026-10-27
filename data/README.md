# Dataset

The raw dataset is not included in this repository. Obtain it from the
[UrbanEV repository](https://github.com/IntelligentSystemsLab/UrbanEV) and
review the upstream license and terms before downloading or using it.

Place the cleaned, aggregated dataset under:

```text
data/raw/20220901-20230228_zone-cleaned-aggregated/
```

The project expects the `charge_1hour/` matrices and the zone-level
`adj.csv`, `distance.csv`, and `zone-information.csv` files in that directory.
See `configs/experiment.yaml` for the expected file names. Raw data files under
`data/raw/` are intentionally excluded from Git.
