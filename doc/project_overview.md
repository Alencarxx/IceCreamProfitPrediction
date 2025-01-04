# Project Overview: Ice Cream Profit Prediction

## 1. Project Description
This project predicts the daily revenue of an ice cream business based on the temperature using:
1. A **simple neural network** built with TensorFlow, and
2. A **Linear Regression model** from scikit-learn for comparison.

The dataset contains two columns:
- **Temperature**: Daily temperature in degrees Celsius.
- **Revenue**: Daily revenue in dollars.

---

## 2. Project Objectives
- Train a neural network to model the relationship between temperature and revenue.
- Compare the predictions of the neural network with a simple linear regression model.
- Provide visual insights into the relationship between temperature and revenue.

---

## 3. Project Structure

IceCreamProfitPrediction/ 
│
├── data/ 
│ └── SalesData.csv # CSV file containing input data 
│ ├── src/ 
│ └── main.py # Main project script 
│ ├── doc/ 
│ └── project_overview.md # Detailed project documentation 
│ ├── README.md # Main project document 
├── README_EN.md # English version of the README 
├── requirements.txt # Project dependencies 
└── .gitignore # Files to be ignored by Git


---

## 4. Detailed Workflow

### 4.1 Data Loading
The dataset is loaded from `data/SalesData.csv` using Pandas. The data contains two columns:
- **Temperature**: Input variable (independent variable).
- **Revenue**: Target variable (dependent variable).

### 4.2 Exploratory Data Analysis (EDA)
- Displays the dataset structure and summary statistics.
- Visualizes the data using a scatter plot to reveal the linear relationship between temperature and revenue.

### 4.3 Data Preparation
The independent (`x_train`) and dependent (`y_train`) variables are extracted from the dataset:
- `x_train`: A 2D array of temperatures.
- `y_train`: A 2D array of revenues.

### 4.4 Neural Network Model
A neural network is created and trained with the following characteristics:
- **Architecture**:
  - One dense layer with 10 neurons (hidden layer).
  - One dense layer with 1 neuron (output layer).
- **Optimization**: Adam optimizer with a learning rate of 0.1.
- **Loss Function**: Mean Squared Error (MSE).
- **Training**: The model is trained for 500 epochs.

### 4.5 Model Evaluation
The loss during training is plotted to observe the model's learning progress. Lower loss values indicate better performance.

### 4.6 Predictions
The trained neural network predicts the revenue for a given temperature. For example:
- Input: `Temperature = 5°C`
- Output: `Predicted Revenue ≈ $X`

### 4.7 Comparison with Linear Regression
A simple linear regression model is trained on the same dataset and its predictions are compared with the neural network. The regression line is plotted alongside the data points for visual comparison.

---

## 5. Key Functions in the Code

### 5.1 `load_data(file_path)`
- **Purpose**: Loads the dataset from a CSV file.
- **Input**: File path of the CSV.
- **Output**: A Pandas DataFrame.

### 5.2 `analyze_data(df)`
- **Purpose**: Performs exploratory data analysis.
- **Input**: Pandas DataFrame.
- **Output**: Displays statistics and generates a scatter plot.

### 5.3 `prepare_data(df)`
- **Purpose**: Prepares the data for training.
- **Input**: Pandas DataFrame.
- **Output**: `x_train` and `y_train` as NumPy arrays.

### 5.4 `create_and_train_model(x_train, y_train)`
- **Purpose**: Creates and trains a neural network model.
- **Input**: Training data (`x_train`, `y_train`).
- **Output**: Trained model.

### 5.5 `evaluate_model(model)`
- **Purpose**: Evaluates and visualizes the model's performance.
- **Input**: Trained model.
- **Output**: A loss curve plot.

### 5.6 `make_predictions(model, temperature)`
- **Purpose**: Makes predictions using the trained model.
- **Input**: Model and temperature value.
- **Output**: Predicted revenue.

### 5.7 `compare_with_sklearn(x_train, y_train)`
- **Purpose**: Trains a linear regression model and compares its predictions with the neural network.
- **Input**: Training data (`x_train`, `y_train`).
- **Output**: A plot showing the regression line and predictions.

---

## 6. Results and Insights
- Both the neural network and linear regression models accurately predict the relationship between temperature and revenue.
- The neural network provides flexibility for capturing more complex relationships if needed.
- Linear regression serves as a baseline for comparison.

---

## 7. Technologies Used
- **Python**: Programming language.
- **TensorFlow**: For the neural network model.
- **scikit-learn**: For the linear regression model.
- **Pandas**: Data loading and manipulation.
- **NumPy**: Array operations.
- **Matplotlib & Seaborn**: Data visualization.

---

## 8. Author
- **Name**: Alencar Porto
- **Date**: 01/04/2025
