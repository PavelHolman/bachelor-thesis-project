# TODO

### Loading and cleaning data

- [x] Load OTE a.s. raw data
    - [x] DA
    - [x] ID
- [x] Load ENTSO-E raw data
    - [x] Generation forecast
        - [x] Solar
        - [x] Wind
    - [x] Load forecast
- [x] Load Open-Meteo raw data
    - [x] Temperature
    - [x] Rain
    - [x] Wind speed
    - [x] Wind direction
    - [x] Wind gusts
    - [x] Solar radiation
- [x] Merge raw data into interim data

### Feature engineering

- [x] Generate rolling ID features
- [x] Generate rolling DA features
- [x] Generate static time features
- [x] Datatype conversion

### Prepare data for training

- [x] item_id and timestamp for AutoGluon dataset
- [x] Split data into training, validation, selection and test sets
- [x] Prepare datasets for AutoGluon
    - [x] DA
    - [x] ID

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
