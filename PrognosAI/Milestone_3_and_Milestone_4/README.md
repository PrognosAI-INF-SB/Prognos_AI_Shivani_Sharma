# PrognosAI: AI-Driven Predictive Maintenance System Using Time-Series Sensor Data
Milestone 3 — Model Development & Evaluation
Prepared by: Shivani Sharma
Dataset: NASA Turbofan Jet Engine Data Set

## Project Overview
PrognosAI is an AI-powered predictive maintenance system designed to forecast the Remaining Useful Life (RUL) of turbofan jet engines using time-series sensor data.
This milestone focuses on developing, training, and evaluating an LSTM model capable of learning degradation patterns and predicting failure before it occurs.

The project leverages NASA’s CMAPSS dataset, which simulates multiple aircraft engines operating under different conditions and fault modes. The main goal is to detect early signs of engine wear and accurately estimate how many cycles remain before failure.

## Milestone 3 Objective
- Build and train a deep learning model (LSTM) for RUL prediction.
- Optimize model parameters for better generalization and higher R² accuracy (>98%).
- Evaluate model performance using regression metrics (RMSE, MAE, R²).
- Save and document preprocessing tools and trained model for future deployment.

## Modules and Libraries Used

| Module | Purpose |
|---------|----------|
| pandas | For reading and processing the CMAPSS dataset into structured DataFrames. |
| numpy | Numerical operations and array manipulation for model input preparation. |
| os | File management and directory operations. |
| joblib | Saving and loading preprocessing scalers and trained models efficiently. |
| scikit-learn (sklearn) | Provides MinMaxScaler for normalization and regression evaluation metrics. |
| tensorflow / keras | Deep learning framework for building and training LSTM models. |
| matplotlib | Visualization of RUL predictions and loss trends. |
| seaborn | Enhanced visualizations for correlation and feature importance analysis. |

## Workflow

### 1. Data Loading and Inspection
The NASA CMAPSS dataset contains multiple sensor readings recorded over time for each engine.
Each engine runs until failure, and the number of operational cycles represents its lifespan.

Key Steps:
- Read training and test datasets using pandas.read_csv().
- Assign column names such as ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 'sensor1', ... 'sensor21'].
- Check for missing or constant-value sensors.

### 2. Data Preprocessing
Preprocessing ensures uniformity and prepares data for model input.

Operations performed:
- Computed Remaining Useful Life (RUL) for each engine:
  ```python
  rul = max_cycle - current_cycle
  ```
- Dropped constant and irrelevant sensors.
- Normalized input features using MinMaxScaler from sklearn.preprocessing.
- Generated time-windowed sequences to capture temporal dependencies:
  ```python
  def sequence_data(df, seq_length=30):
      sequences = []
      for i in range(len(df) - seq_length):
          seq_x = df.iloc[i:i+seq_length].values
          seq_y = df.iloc[i+seq_length]['RUL']
          sequences.append((seq_x, seq_y))
      return sequences
  ```

### 3. Model Development – LSTM Architecture
The model uses Long Short-Term Memory (LSTM) layers to handle time-series dependencies.

Architecture:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(128, input_shape=(sequence_length, num_features), return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
```

Compilation:
```python
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

Training:
```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val), 
                    epochs=100, 
                    batch_size=64, 
                    callbacks=[early_stop, reduce_lr])
```

### 4. Model Evaluation
After training, the model is evaluated on test data using regression metrics:

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
```

Performance Example:

| Metric | Value |
|--------|--------|
| R² Score | 0.983 |
| RMSE | 0.018 |
| MAE | 0.017 |

The high R² indicates the model accurately predicts RUL trends with minimal deviation.

### 5. Model Saving
For reuse and deployment, both model and scalers are stored:

```python
model.save("optimized_lstm_final.keras")
joblib.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, "final_scalers.pkl")
```

These files are later used in the PrognosAI Streamlit dashboard for visualization and predictions.

## Results and Visualization
- Training vs Validation Loss plots show stable convergence.
- Predicted vs Actual RUL demonstrates high correlation with minimal error.
- The model successfully generalizes across different engine units and operating conditions.

## Conclusion
Milestone 3 establishes the core LSTM prediction engine of PrognosAI.
By leveraging time-series learning, the model effectively captures degradation patterns and achieves high predictive accuracy.
This trained model and its preprocessing pipeline form the foundation for Milestone 4 (Dashboard and Visualization) and subsequent deployment stages.

## Key Takeaways
- Built scalable LSTM architecture for sequence learning.
- Achieved >98% R² score with optimized hyperparameters.
- Established reusable preprocessing and model storage workflow.
- Ready for integration into PrognosAI’s real-time inference dashboard.

## References
- NASA CMAPSS Turbofan Degradation Dataset
- TensorFlow and Keras Documentation
- Scikit-learn User Guide

© 2025 Shivani Sharma — PrognosAI Project
