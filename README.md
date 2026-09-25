# Retail Demand Simulator

Advanced synthetic retail demand simulation and intelligence system.

## Technology

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Joblib

## Features

### Data simulation

The system generates:

- Customers
- Products
- Stores
- Daily sales
- Inventory
- Promotions
- Competitor prices
- Festivals
- Weather
- Seasonal demand

### Relationships

The simulation includes relationships between:

- Store traffic and sales
- Product popularity and sales
- Customer preferences and products
- Price and demand
- Competitor price and demand
- Promotions and demand
- Festivals and demand
- Weather and demand
- Weekend and demand
- Inventory and stockouts
- Seasonality and demand

## ML

The forecasting model uses:

- Previous day demand
- Previous 7-day demand
- Previous 14-day demand
- Previous 28-day demand
- 7-day rolling demand
- 28-day rolling demand
- Promotion
- Discount
- Competitor price gap
- Weather
- Temperature
- Month
- Day of week
- Weekend
- Store
- Product

Model:

GradientBoostingRegressor

## Customer segmentation

KMeans clustering is applied to:

- Purchase frequency
- Units purchased
- Revenue
- Average order value
- Discount behavior

Segments:

- Low Value
- Regular
- High Value
- VIP

## Dashboard

Pages:

/ 
/eda
/segments
/forecast
/inventory

## Installation

Create environment:

python -m venv venv

Activate:

Windows:

venv\Scripts\activate

Install:

python -m pip install -r requirements.txt

## Generate data

python simulate.py

## Run data pipeline

python pipeline.py

## Train model

python train.py

## Start Flask

python run.py

Open:

http://127.0.0.1:5000