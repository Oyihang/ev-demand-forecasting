# Dataset

The raw dataset is not included in this repository. Obtain it from the
[UrbanEV repository](https://github.com/IntelligentSystemsLab/UrbanEV) and
review the upstream license and terms before downloading or using it.

Place the cleaned, aggregated dataset under:

```text
data/raw/20220901-20230228_zone-cleaned-aggregated/
```

The project expects the `charge_1hour/` matrices and the zone-level
`adj.csv`, `distance.csv`, `zone-information.csv`, and `weather_central.csv`
files in that directory. The weather table must contain `time`, `T`, `P0`, `P`,
`U`, `Td`, and `nRAIN`, with complete values for target timestamps.
See `configs/experiment.yaml` for the expected file names. Raw data files under
`data/raw/` are intentionally excluded from Git.
