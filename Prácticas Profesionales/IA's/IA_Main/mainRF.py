import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import math


def ajustar_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 3, 5]
    }

    tscv = TimeSeriesSplit(n_splits=3)
    modelo = RandomForestRegressor(random_state=42, n_jobs=-1)

    grid = GridSearchCV(modelo, param_grid, cv=tscv, scoring='neg_mean_absolute_error', verbose=2)
    grid.fit(X_train, y_train)

    print(f"Mejores hiperparámetros encontrados: {grid.best_params_}")
    return grid.best_estimator_


def cargar_y_preparar_datos(ruta_archivo, target='valor (kWh)', features=None):
    df = pd.read_excel(ruta_archivo)
    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
    df.set_index('fecha_hora', inplace=True)

    df['hora'] = df.index.hour
    df['dia_semana'] = df.index.dayofweek
    df['mes'] = df.index.month
    df['fin_de_semana'] = (df['dia_semana'] >= 5).astype(int)

    if features is not None:
        features = [f for f in features if f in df.columns]
    else:
        features = df.columns.drop(target)

    X = df[features]
    y = df[target]
    return df, X, y, features


def calcular_metricas(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def graficar_comparacion(train_df, test_df, features):
    variables = ['y'] + features
    n = len(variables)
    filas = math.ceil(n / 2)

    fig, axs = plt.subplots(filas, 2, figsize=(16, 3 * filas), sharex=True)
    axs = axs.ravel()

    for i, var in enumerate(variables):
        axs[i].plot(train_df.index, train_df[var], label='Entrenamiento', color='blue', alpha=0.6)
        axs[i].plot(test_df.index, test_df[var], label='Prueba', color='orange', alpha=0.6)
        axs[i].set_ylabel(var)
        axs[i].legend()

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    axs[min(len(axs) - 1, i)].set_xlabel('Fecha')
    fig.suptitle('Comparación entre Entrenamiento y Prueba', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def main():
    ruta = r"C:\Prácticas Profesionales\JSON\Resultado_Homogenizado.xlsx"
    target = 'valor (kWh)'
    features = ['eToday (kWh)', 'eTotal (kWh)', 'power (kW)',
                'wind_speed_100m', 'wind_speed_10m', 'wind_direction_10m',
                'relative_humidity', 'hora', 'dia_semana', 'mes', 'fin_de_semana']

    df, X, y, features = cargar_y_preparar_datos(ruta, target, features)

    periodo_test = '30D'
    test_data = df.loc[df.index >= df.index.max() - pd.Timedelta(periodo_test)]
    train_data = df.loc[df.index < test_data.index.min()]

    X_train = X.loc[train_data.index]
    y_train = y.loc[train_data.index]
    X_test = X.loc[test_data.index]
    y_test = y.loc[test_data.index]

    # Entrenar modelo con hiperajuste
    modelo_rf = ajustar_random_forest(X_train, y_train)

    # Predicción
    y_pred = modelo_rf.predict(X_test)

    # Métricas
    rmse, mae, r2 = calcular_metricas(y_test, y_pred)
    print(f"\nEvaluación modelo Random Forest (hiperajustado) para {periodo_test}:")
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

    dias_totales = (df.index.max() - df.index.min()).days
    print(f"Días totales en el dataset: {dias_totales}")

    # Preparar dataframes con 'y'
    train_df = train_data.copy()
    test_df = test_data.copy()
    train_df['y'] = y_train
    test_df['y'] = y_test

    graficar_comparacion(train_df, test_df, features)

    # Coincidencias dentro de tolerancia
    tolerancia = 0.05
    coincidencias = np.isclose(y_test, y_pred, rtol=tolerancia)
    porcentaje = 100 * np.sum(coincidencias) / len(y_test)

    print(f"Coincidencias dentro del {tolerancia*100:.0f}% de tolerancia: {np.sum(coincidencias)} de {len(y_test)}")
    print(f"Porcentaje de coincidencia aproximada: {porcentaje:.2f}%")

    print(f"Periodo de entrenamiento: {train_data.index.min().date()} a {train_data.index.max().date()}")
    print(f"Periodo de prueba: {test_data.index.min().date()} a {test_data.index.max().date()}")
    print(f"Total de datos: {len(df)}")
    print(f"Datos de entrenamiento: {len(train_data)}, Datos de prueba: {len(test_data)}")


if __name__ == '__main__':
    main()
