# End-to-End Car Price Prediction

## Project Overview  
A machine learning solution to predict car resale prices based on features such as mileage, age, fuel type, etc. The project includes data cleaning, feature engineering, model training, and a web interface streamlit for real-time predictions.

## Features  
- Data preprocessing (handling missing values, duplicates, encoding categorical features)  
- Feature engineering
- Model training — using CatBoost 
- Serialization of trained model for inference (e.g. `.sav` or `.joblib`)  
- Web app interface for users to enter car attributes and get price predictions  

## Installation  

```bash
# clone the repo  
git clone https://github.com/abdelrahmansolii/End-to-End-Car_price_prediction.git  
cd End-to-End-Car_price_prediction  

# create virtual environment (optional but recommended)  
python -m venv venv  
source venv/bin/activate  # (on Windows: venv\Scripts\activate)

# install dependencies  
pip install -r requirements.txt  
