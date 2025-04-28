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

- [x] Train AutoGluon models
    - [x] DA known covariates
    - [x] ID known covariates
    - [x] TimeSeries and Tabular models
    - [x] Evaluation metrics
        - [x] MAE, RMSE
        - [ ] WMAE
    - [x] Hyperparameter tuning
    - [x] Model selection

### Evaluation

- [x] Backtesting
- [x] Evaluate each trained model
    - [x] Plot predictions vs. actuals
    - [x] Plot error distribution
    - [ ] Plot metrics
- [ ] Retrain best model on full training set and repeat evaluation on test set

### Add some EDA

- [ ] EDA
- [ ] DA as ID
- [ ] Feature importance
