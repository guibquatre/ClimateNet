## SAM BLOUIN 20035269

### Requirements
```bash
conda env create -f conda_env.yaml
conda activate climate
conda env list
```

## Run code
```bash
python example.py
```

| Files                       |                                  directory location or meaning |
|:----------------------------|---------------------------------------------------------------:|
| data                        |        classification-of-extreme-weather-events-udem directory |
| grid_searchs                | files for hyperparameter tuning (not used for your final test) |
| grid_search_rf_final.joblib |                               model result from grid_search.py |
| conda_env.yml               |                 file for requirements, see bash commands above |