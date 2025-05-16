# Forecasting energy costs in day-ahead and intraday markets

This repository contains the source code, datasets, and experimental results related to a bachelor's thesis focused on
forecasting day-ahead and intraday electricity prices in the Czech market operated by OTE a.s. The project investigates
and compares various forecasting techniques, including classical statistical models, machine learning algorithms, and
deep learning architectures. The core of the project is the development of a forecasting model that predicts electricity
prices in the Czech market.

## Repository Structure

- `data/`: Contains the raw and processed datasets used for training and testing the forecasting models.
- `models/`: Contains the models saved after training, and the results of their predictions.
- `notebooks/`: Jupyter notebooks used for data analysis, model training, and evaluation.
- `src/`: Source code for data fetching, preprocessing, model training, and evaluation.
- `requirements.txt`: List of Python packages required to run the code.
- `utils.py`: Loads the environment variables and initializes the repository structure if needed.
- `README.md`: This file.

## Setup Instructions

To set up the environment and install the required packages, follow these steps:

```commandline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

In case you want to fetch new data, you need
to [get](https://uat-transparency.entsoe.eu/content/static_content/Static%20content/web%20api/how_to_get_security_token.html)
an API key for the ENTSO-E API and provide it as `ENTSOE_API_KEY`environment variable; otherwise part of the data will
be missing.
