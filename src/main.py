# -*- coding: utf-8 -*-
"""
Projeto: Previsão de Lucros (Sorvetes)
Autor: Alencar Porto
Data: 04/01/2025

Este script prevê o lucro diário de um negócio de sorvetes com base na temperatura.
"""

# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.linear_model import LinearRegression
import os

# Configurações gerais
plt.style.use('seaborn')

# Caminho do CSV
DATA_PATH = os.path.join("..", "data", "SalesData.csv")

def main():
    # Etapa 1: Importação da base de dados
    sales_df = load_data(DATA_PATH)

    # Etapa 2: Análise exploratória
    analyze_data(sales_df)

    # Etapa 3: Preparação dos dados
    x_train, y_train = prepare_data(sales_df)

    # Etapa 4: Criação e treinamento do modelo
    model = create_and_train_model(x_train, y_train)

    # Etapa 5: Avaliação do modelo
    evaluate_model(model)

    # Etapa 6: Previsões
    make_predictions(model, temperature=5)

    # Etapa 7: Comparação com modelo sklearn
    compare_with_sklearn(x_train, y_train)

def load_data(file_path):
    """Carregar dados do arquivo CSV."""
    try:
        data = pd.read_csv(file_path)
        print(f"Dados carregados com sucesso de {file_path}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo {file_path} não encontrado. Verifique o caminho!")

def analyze_data(df):
    """Análise exploratória dos dados."""
    print(df.info())
    print(df.describe())
    sns.scatterplot(x=df['Temperature'], y=df['Revenue'])
    plt.title('Scatter Plot: Temperatura vs Receita')
    plt.show()

def prepare_data(df):
    """Preparação dos dados para treinamento."""
    x_train = df['Temperature'].values.reshape(-1, 1)
    y_train = df['Revenue'].values.reshape(-1, 1)
    return x_train, y_train

def create_and_train_model(x_train, y_train):
    """Criação e treinamento da rede neural."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=10, input_shape=[1]),
        tf.keras.layers.Dense(units=1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')
    history = model.fit(x_train, y_train, epochs=500, verbose=0)
    model.history = history
    return model

def evaluate_model(model):
    """Avaliação do progresso do treinamento."""
    plt.plot(model.history.history['loss'])
    plt.title('Progressão da Perda do Modelo Durante o Treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Perda de Treinamento')
    plt.legend(['Perda de Treinamento'])
    plt.show()

def make_predictions(model, temperature):
    """Fazer previsões com o modelo treinado."""
    revenue = model.predict(np.array([[temperature]]))
    print(f'Previsão de Receita para {temperature}°C = {revenue[0][0]:.2f} dólares')

def compare_with_sklearn(x_train, y_train):
    """Comparação com a Regressão Linear do sklearn."""
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    
    # Gráfico com a reta da regressão linear
    plt.scatter(x_train, y_train, color='gray')
    plt.plot(x_train, regressor.predict(x_train), color='red')
    plt.ylabel('Receita [dólares]')
    plt.xlabel('Temperatura [°C]')
    plt.title('Receita Gerada vs. Temperatura @Sorveteria')
    plt.show()

    # Previsões com o modelo treinado, com a temperatura de 5 graus
    revenue = regressor.predict(np.array([[5]]))
    print(f'Previsão de Receita usando Regressão Linear = {revenue[0][0]:.2f} dólares')

if __name__ == "__main__":
    main()
