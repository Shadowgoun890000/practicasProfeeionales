import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Cargar el archivo y configurar la columna de fecha
file_path = r"C:\Prácticas Profesionales\JSON\Resultado_Homogenizado.xlsx"
data = pd.read_excel(file_path)

data['fecha_hora'] = pd.to_datetime(data['fecha_hora'])
data.set_index('fecha_hora', inplace=True)

# Selección de la columna de objetivo y características específicas para Saltillo
target = 'valor (kWh)'

# Características meteorológicas seleccionadas que se alinean con Saltillo
features = [ 
    'eToday (kWh)', 'eTotal (kWh)', 'power (kW)',
    'wind_speed_100m', 'wind_speed_10m', 
    'wind_direction_10m',"relative_humidity"
]

# Asegurarse de que las características estén presentes en los datos
features = [col for col in features if col in data.columns]

# Crear conjunto de entrenamiento y prueba
train, test = train_test_split(data, test_size=0.2, shuffle=False)

# Extraer las series temporales de entrenamiento y prueba
train_series = train[target]
test_series = test[target]

# Análisis Exploratorio de Datos (EDA)
print("Estadísticas descriptivas de las variables:")
print(data.describe())

# Distribución de la generación de energía y consumo
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
sns.histplot(data[target], kde=True)
plt.title(f'Distribución de {target}')

# Correlación entre las variables
correlation_matrix = data[features + [target]].corr()
plt.subplot(2, 1, 2)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación')
plt.tight_layout()
plt.show()

# Crear conjunto de entrenamiento y prueba
train, test = train_test_split(data, test_size=0.2, shuffle=False)

# Extraer las series temporales de entrenamiento y prueba
train_series = train[target]
test_series = test[target]

# Modelos a probar
# 1. ARIMA (utiliza solo la serie temporal univariante)
arima_model = ARIMA(train_series, order=(5,1,0))
arima_model_fit = arima_model.fit()
arima_pred = arima_model_fit.forecast(steps=len(test_series))

# 2. Prophet (modelo de series temporales con estacionalidad)
prophet_data = train.reset_index().rename(columns={'fecha_hora': 'ds', target: 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_data)
future = prophet_model.make_future_dataframe(periods=len(test_series), freq='5T')
prophet_forecast = prophet_model.predict(future)['yhat'].iloc[-len(test_series):]

# 3. Random Forest Regressor (modelo supervisado con características adicionales)
X_train, X_test = train[features], test[features]
y_train, y_test = train[target], test[target]
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Función para calcular métricas de evaluación
def calcular_métricas(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# Calcular métricas para cada modelo
arima_metrics = calcular_métricas(test_series, arima_pred)
prophet_metrics = calcular_métricas(test_series, prophet_forecast)
rf_metrics = calcular_métricas(y_test, rf_pred)

# Resultados de las métricas
print("Métricas de ARIMA - RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(*arima_metrics))
print("Métricas de Prophet - RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(*prophet_metrics))
print("Métricas de Random Forest - RMSE: {:.2f}, MAE: {:.2f}, R2: {:.2f}".format(*rf_metrics))

# Graficar los resultados
plt.figure(figsize=(12, 8))

# Gráfico de las series temporales reales y las predicciones
plt.subplot(3, 1, 1)
plt.plot(test_series.index, test_series, label='Real', color='black')
plt.plot(test_series.index, arima_pred, label='ARIMA', color='blue')
plt.title('Predicción de ARIMA')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(test_series.index, test_series, label='Real', color='black')
plt.plot(test_series.index, prophet_forecast, label='Prophet', color='green')
plt.title('Predicción de Prophet')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(test_series.index, test_series, label='Real', color='black')
plt.plot(test_series.index, rf_pred, label='Random Forest', color='red')
plt.title('Predicción de Random Forest')
plt.legend()

plt.tight_layout()
plt.show()


# Verificar correlaciones con la variable objetivo
correlaciones = data[features + [target]].corr()[target]
print("Correlaciones con la variable objetivo:")
print(correlaciones)

# Verificar constantes (varianza igual a 0)
varianzas = data[features].var()
constantes = varianzas[varianzas == 0].index.tolist()
print("Variables constantes:", constantes)