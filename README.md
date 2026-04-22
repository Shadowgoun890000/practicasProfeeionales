# ⚡ Sistema de Visualización y Predicción Energética con Machine Learning

Proyecto académico orientado al análisis, integración, modelado y visualización de datos energéticos y climáticos de un sistema fotovoltaico, utilizando técnicas de **Machine Learning** para la predicción del comportamiento energético del sistema.

---

## 📌 Descripción general

Este proyecto integra múltiples fuentes de datos para construir un pipeline completo de análisis energético:

- extracción de datos de generación fotovoltaica desde archivos JSON de **Growatt**,
- extracción y procesamiento de datos climáticos,
- homogenización temporal de ambas fuentes,
- evaluación de múltiples estrategias de validación para modelos predictivos,
- selección de un modelo final basado en **Random Forest**,
- implementación de una aplicación interactiva en **Python Shiny** para visualización histórica y predicción.

El sistema fue diseñado como una plataforma experimental para comparar distintos modelos de series temporales y aprendizaje supervisado aplicados al comportamiento energético de un sistema fotovoltaico.

---

## 🎯 Objetivo del proyecto

Desarrollar y evaluar un sistema de predicción energética basado en variables climáticas, temporales y operativas de un sistema fotovoltaico, con el fin de estimar el comportamiento futuro de la señal energética registrada y visualizar los resultados mediante una aplicación interactiva.

---

## 📂 Fuentes de datos

El proyecto utiliza dos fuentes principales de información.

### 1. Datos de generación fotovoltaica
Provienen de archivos JSON exportados desde **Growatt**. Estos archivos incluyen variables como:

- `nominalPower`
- `eToday`
- `eTotal`
- `invPacData`

La clave `invPacData` contiene una serie temporal con registros cada **5 minutos**.

### 2. Datos climáticos
Provienen de archivos JSON con observaciones meteorológicas, incluyendo variables como:

- `air_temp`
- `relative_humidity`
- `ghi`
- `dni`
- `gti`
- `wind_speed_10m`
- `wind_direction_10m`
- `period_end`

Los datos climáticos también fueron trabajados a resolución temporal de **5 minutos**.

---

## 🧠 Variable objetivo

La variable objetivo utilizada en el modelado corresponde a la señal energética del sistema fotovoltaico registrada en el dataset homogenizado.

> **Importante:**  
> El modelo final predice el comportamiento energético o **generación del sistema fotovoltaico**, no el consumo facturable de la vivienda.

---

## 🗂️ Estructura del proyecto

```text
practicasProfeeionales/
│
├── .venv/
├── Prácticas Profesionales/
│   ├── .idea/
│   ├── app/
│   ├── data/
│   ├── Dataframe/
│   ├── IA's/
│   │   ├── IA's results/
│   │   ├── IA's Tests/
│   │   └── IA_Main/
│   │       └── train_random_forest_final.py
│   ├── JSON/
│   ├── ml/
│   ├── models/
│   ├── Templates/
│   └── App.py
│
├── .gitattributes
└── .gitignore
```

---

## 📁 Descripción de carpetas

### `app/`
Contiene la lógica principal de la aplicación en **Python Shiny**.

Archivos principales:
- `config.py` → configuración global de la app
- `main.py` → inicialización de la app
- `server.py` → lógica reactiva del servidor
- `ui.py` → construcción de la interfaz gráfica

### `data/`
Contiene utilidades relacionadas con la carga y transformación de datos.

Archivos principales:
- `loader.py` → carga del dataset principal
- `features.py` → generación de variables derivadas
- `merge.py` → integración de datos

### `Dataframe/`
Incluye scripts de preprocesamiento, análisis exploratorio y construcción del dataset final.

Archivos principales:
- `Clima.py`
- `daily data energy generation.py`
- `merge.py`
- `series_temporales.py`
- `Código_Fuente_CostosdeConsumo.py`

También contiene figuras generadas para análisis temporal:
- `analisis_completo_estacionalidad.png`
- `serie_temporal_patron_diario.png`
- `serie_temporal_tesis.png`

### `IA's/`
Contiene el bloque principal de experimentación y entrenamiento.

#### `IA's Tests/`
Incluye scripts de evaluación con distintas estrategias de validación y comparación de modelos.

#### `IA's results/`
Contiene resultados generados por las pruebas, como métricas, tablas y gráficas.

#### `IA_Main/`
Incluye el script de entrenamiento final del modelo seleccionado:
- `train_random_forest_final.py`

### `JSON/`
Almacena los archivos fuente crudos:
- datos climáticos,
- datos Growatt,
- y archivos derivados intermedios.

### `ml/`
Contiene la lógica específica del modelo de Machine Learning.

Archivos principales:
- `schema.py` → definición de variables de entrada del modelo
- `model_loader.py` → carga del modelo serializado
- `predictor.py` → preparación de entrada para inferencia
- `forecast.py` → generación de horizonte futuro
- `economics.py` → estimación económica simple

### `models/`
Contiene los modelos serializados entrenados, por ejemplo:
- `random_forest_60d.joblib`

---

## 🔄 Pipeline del proyecto

El flujo general del proyecto es el siguiente:

1. **Extracción de datos**
   - lectura de archivos JSON climáticos,
   - lectura de archivos JSON de Growatt.

2. **Preprocesamiento**
   - transformación de timestamps,
   - limpieza y selección de variables,
   - unificación de frecuencia temporal.

3. **Homogenización**
   - integración de clima y generación en un solo dataset,
   - construcción de `Resultado_Homogenizado.xlsx`.

4. **Ingeniería de características**
   - variables temporales:
     - hora
     - día de la semana
     - fin de semana
     - mes
     - estación

5. **Entrenamiento y evaluación**
   - comparación de varios modelos:
     - ARIMA
     - Prophet
     - Random Forest

6. **Estrategias de validación**
   - K-Fold temporal
   - TimeSeriesSplit
   - Train-Test Split por periodos
   - Train-Test Variable por periodos

7. **Selección del modelo final**
   - Random Forest fue seleccionado por su mejor desempeño global.

8. **Despliegue**
   - integración del modelo entrenado en una aplicación Shiny.

---

## 🤖 Modelos evaluados

Durante la fase experimental se evaluaron tres enfoques principales.

### ARIMA
Modelo clásico de series temporales, útil como referencia base.

### Prophet
Modelo orientado a capturar tendencia y estacionalidad con regresores externos.

### Random Forest
Modelo de aprendizaje supervisado basado en árboles de decisión, seleccionado como modelo final por su robustez y mejor desempeño general.

---

## 🏆 Modelo final seleccionado

El modelo final del proyecto es un **Random Forest Regressor** entrenado con variables:

### Variables operativas del sistema
- `eToday (kWh)`
- `eTotal (kWh)`
- `power (kW)`

### Variables meteorológicas
- `air_temp`
- `relative_humidity`
- `ghi`
- `dni`
- `gti`
- `wind_speed_10m`
- `wind_direction_10m`

### Variables temporales
- `hora`
- `dia_semana`
- `es_fin_semana`
- `mes`
- `estacion`

El modelo entrenado se guarda en formato `.joblib` dentro de la carpeta `models/`.

---

## 🖥️ Aplicación

La aplicación permite:

- visualizar datos históricos energéticos,
- visualizar variables climáticas,
- generar predicciones futuras con el modelo entrenado,
- mostrar métricas del modelo cargado,
- estimar una traducción económica simple del horizonte predicho.

### Módulos principales de la app
- **Resumen**
- **Histórico energético**
- **Clima**
- **Predicción**
- **Información del modelo**

---

## ▶️ Ejecución del proyecto

### 1. Activar entorno virtual

```bash
source .venv/bin/activate
```

### 2. Entrenar el modelo final

```bash
python "Prácticas Profesionales/IA's/IA_Main/train_random_forest_final.py"
```

### 3. Ejecutar la aplicación

```bash
python "Prácticas Profesionales/App.py"
```

---

## 📦 Dependencias principales

Este proyecto utiliza principalmente:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `statsmodels`
- `prophet`
- `joblib`
- `shiny`
- `openpyxl`

---

## 📊 Resultados experimentales

Las estrategias evaluadas mostraron que:

- **ARIMA** presenta limitaciones para reproducir adecuadamente la variabilidad observada.
- **Prophet** puede capturar parte de la estructura temporal, pero degrada su desempeño en horizontes largos.
- **Random Forest** mostró el mejor equilibrio entre error bajo, estabilidad y capacidad de ajuste.

Por esta razón, **Random Forest** fue seleccionado como el modelo final para la app.

---

## ⚠️ Limitaciones

- El modelo actual trabaja sobre la señal energética o **generación del sistema fotovoltaico**.
- No predice directamente el **consumo facturable** de la vivienda.
- La estimación económica implementada en la app es una aproximación simple.
- Para modelar consumo real o recibo eléctrico, sería necesario construir un dataset específico con variables objetivo de consumo facturable.

---

## 🚀 Trabajo futuro

- incorporar una conversión energética más rigurosa de **W a kWh** en la app,
- integrar análisis bimestral basado en recibos CFE,
- construir un modelo específico para consumo facturable,
- mejorar la visualización y modularización de la aplicación,
- agregar exportación de resultados y reportes.

---

## 📍 Estado del proyecto

Proyecto académico funcional en fase de consolidación, con modelo final entrenado e integración en aplicación interactiva para visualización y predicción energética.

---

## 👤 Autor

**Jorge Antonio Carbajal Aguilar**

---

## 🎓 Contexto académico

Proyecto desarrollado en la  
**Universidad Autónoma de Coahuila**  
**Facultad de Sistemas**
