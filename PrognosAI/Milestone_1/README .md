CMAPSS Sensor Data Preparation and Feature Engineering
Overview 
This project focuses on preparing and engineering features from sensor data collected in aircraft engines for Remaining Useful Life (RUL) prediction. The goal is to clean and preprocess the CMAPSS dataset, create rolling window sequences capturing temporal dependencies, and compute accurate RUL targets for model training.

Features 
*Handles missing sensor values through interpolation to maintain data quality. 
*Standardizes sensor data for consistent scale across features.
*Generates rolling windows that capture time-series behavior for robust modeling.
*Computes precise RUL values aligned with each engine cycle.

Includes data integrity validations to ensure clean and reliable inputs.

How It Works

The pipeline loads raw sensor data, fills missing values, and normalizes features. Next, it creates fixed-length rolling sequences to model the temporal progression of engine degradation. The Remaining Useful Life for each data point is calculated as the distance to the end-of-life cycle. This processed data is then ready for training machine learning models for predictive maintenance.

Project Structure

Data loading and preprocessing scripts Rolling window generation module RUL computation function Validation checks to ensure data accuracy and integrity

Usage Intended as a preprocessing step before predictive modeling, this project sets a solid foundation for effective Remaining Useful Life prediction through thorough data preparation and feature engineering.

Future Work Plans include extending support for multiple engine units, incorporating additional advanced statistical and signal processing features, and integrating with deep learning models for enhanced prognostics.
