# TODO

### Loading and cleaning data

- [ ] Load OTE a.s. raw data
    - [ ] DA
    - [ ] ID
- [ ] Load ENTSO-E raw data
    - [ ] Generation forecast
        - [ ] Total
        - [ ] Solar
        - [ ] Wind
    - [ ] Load forecast
- [ ] Load Open-Meteo raw data
    - [ ] Temperature
    - [ ] Rain
    - [ ] Wind speed
    - [ ] Wind direction
    - [ ] Wind gusts
    - [ ] Solar radiation
- [ ] Cleanup raw data into interim data

### Feature engineering

- [ ] Generate rolling ID features
- [ ] Generate rolling DA features
- [ ] Generate static time features
- [ ] Datatype conversion

### Prepare data for training

- [ ] item_id and timestamp for AutoGluon dataset
- [ ] Split data into training, validation, selection and test sets
- [ ] Prepare datasets for AutoGluon
    - [ ] DA
    - [ ] ID

### Train models

- [ ] Train AutoGluon models
    - [ ] DA known covariates
    - [ ] ID known covariates
    - [ ] TimeSeries and Tabular models
    - [ ] Evaluation metrics
        - [ ] MAE, MAPE, RMSE
        - [ ] Maximum error
        - [ ] Error percentiles
    - [ ] Hyperparameter tuning
    - [ ] Feature selection
    - [ ] Model selection

### Evaluation

- [ ] Evaluate each trained model
    - [ ] Evaluate single metric and rolling metrics
        - [ ] MAE, MAPE, RMSE
        - [ ] Maximum error
        - [ ] Error percentiles
    - [ ] Evaluate on selection set
    - [ ] Plot predictions vs. actuals
    - [ ] Plot feature importance
    - [ ] Plot error distribution
- [ ] Retrain best model on full training set and repeat evaluation on test set
