# ⚡ Sistema de Visualización y Predicción de Generación Energética con Machine Learning

Proyecto académico orientado al análisis, integración, modelado y visualización de datos energéticos y climáticos de un sistema fotovoltaico, utilizando técnicas de **Machine Learning** para la predicción de la **generación energética** del sistema.

---

## 📌 Descripción general

Este proyecto integra múltiples fuentes de datos para construir un pipeline completo de análisis y predicción de generación fotovoltaica:

- extracción de datos de generación fotovoltaica desde archivos JSON de **Growatt**,
- extracción y procesamiento de datos climáticos,
- homogenización temporal de ambas fuentes,
- generación de variables temporales derivadas,
- evaluación de estrategias de validación para modelos predictivos,
- selección de un modelo final basado en **Random Forest**,
- implementación de una aplicación interactiva en **Python Shiny** para visualización histórica, consulta climática y predicción.

El sistema fue diseñado como una plataforma académica y experimental para el estudio del comportamiento de la **generación de energía** de un sistema fotovoltaico, así como para la consulta de resultados mediante una interfaz gráfica funcional.

---

## 🎯 Objetivo del proyecto

Desarrollar y evaluar un sistema de predicción de la **generación de energía** de un sistema fotovoltaico a partir de variables históricas energéticas, meteorológicas y temporales, con el fin de estimar el comportamiento futuro de la señal registrada y visualizar los resultados mediante una aplicación interactiva.

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
> El modelo y la aplicación están orientados a la **generación energética del sistema fotovoltaico**.  
> No se trata de un modelo de consumo facturable de la vivienda ni de estimación directa del recibo eléctrico.

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
│   ├── static/
│   └── App.py
│
├── .gitattributes
└── .gitignore
```

La aplicación principal se ejecuta desde `App.py`, el cual importa el objeto `app` desde `app.main`.

---

## 📁 Descripción de carpetas

### `app/`
Contiene la lógica principal de la aplicación en **Python Shiny**.

Archivos principales:
- `config.py` → configuración global de la app, rutas, columnas y constantes
- `main.py` → carga de datos, preparación de features e inicialización de la app
- `server.py` → lógica reactiva del servidor, filtros, tablas, gráficas, predicción y exportaciones
- `ui.py` → construcción de la interfaz gráfica
- `static/` → recursos estáticos como CSS y JavaScript para estilos e interacción

### `data/`
Contiene utilidades relacionadas con la carga y transformación de datos.

Archivos principales:
- `loader.py` → carga del dataset principal desde `Resultado_Homogenizado.xlsx`
- `features.py` → generación de variables temporales derivadas
- `merge.py` → integración de datos en etapas previas del proyecto

### `Dataframe/`
Incluye scripts de preprocesamiento, análisis exploratorio y construcción del dataset final.

Archivos principales:
- `Clima.py`
- `daily data energy generation.py`
- `merge.py`
- `series_temporales.py`
- `Código_Fuente_CostosdeConsumo.py`

También contiene figuras generadas para análisis temporal, utilizadas como apoyo durante la etapa exploratoria.

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
- archivos derivados intermedios,
- y el archivo homogenizado final utilizado por la aplicación.

### `ml/`
Contiene la lógica específica del modelo de Machine Learning.

Archivos principales:
- `schema.py` → definición de variables de entrada del modelo
- `model_loader.py` → carga del modelo serializado
- `predictor.py` → construcción de entradas para inferencia
- `forecast.py` → generación de horizonte futuro y pronóstico con el modelo
- `economics.py` → módulo auxiliar de estimación económica simple, no integrado actualmente en el flujo principal de la interfaz

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
   - transformación de marcas temporales,
   - limpieza y selección de variables,
   - unificación de frecuencia temporal.

3. **Homogenización**
   - integración de clima y generación en un solo dataset,
   - construcción del archivo `Resultado_Homogenizado.xlsx`.

4. **Ingeniería de características**
   - variables temporales:
     - `hora`
     - `dia_semana`
     - `es_fin_semana`
     - `mes`
     - `estacion`

5. **Entrenamiento y evaluación**
   - comparación de varios modelos y enfoques de validación.

6. **Selección del modelo final**
   - **Random Forest** fue seleccionado como modelo principal por su desempeño global.

7. **Despliegue**
   - integración del modelo entrenado en una aplicación Shiny,
   - consulta de datos históricos,
   - visualización climática,
   - generación de predicciones,
   - exportación de resultados.

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

El modelo final del proyecto es un **Random Forest Regressor**, integrado en la aplicación como modelo principal de predicción. El sistema también contempla un modo provisional basado en patrón histórico reciente cuando el modelo serializado no está disponible o cuando ocurre una falla durante la inferencia.

### Variables utilizadas por el modelo

De acuerdo con el esquema definido para inferencia, el modelo emplea variables como:

#### Variables energéticas
- `eToday (kWh)`
- `eTotal (kWh)`
- `power (kW)`

#### Variables meteorológicas
- `air_temp`
- `relative_humidity`
- `ghi`
- `dni`
- `gti`
- `wind_speed_10m`
- `wind_direction_10m`

#### Variables temporales
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
- mostrar el estado del modelo cargado,
- exportar resultados en formato CSV.

### Tecnologías utilizadas en la aplicación

- **Python** como lenguaje principal
- **Shiny for Python** para la interfaz y la lógica reactiva
- **Plotly** para gráficas interactivas
- **Matplotlib** para algunas visualizaciones complementarias
- **Pandas** para procesamiento tabular y manipulación de datos
- **shinywidgets** para integrar componentes interactivos en la interfaz

### Módulos principales de la app

#### **Resumen**
Presenta indicadores generales del conjunto de datos:
- total de registros,
- fecha inicial,
- fecha final,
- valor máximo.

#### **Histórico energético**
Permite:
- filtrar por rango de fechas,
- filtrar por rango horario,
- seleccionar la vista de energía:
  - serie completa,
  - promedio diario,
  - máximo diario,
  - últimos 7 días,
  - últimos 30 días,
- consultar tabla de datos,
- visualizar gráfica interactiva,
- descargar resultados filtrados en CSV.

#### **Clima**
Permite:
- filtrar por rango de fechas,
- seleccionar una variable climática,
- consultar la tabla correspondiente,
- visualizar la serie temporal de la variable,
- descargar resultados filtrados en CSV.

#### **Predicción**
Permite:
- definir el horizonte de predicción,
- seleccionar la cantidad de días históricos a mostrar,
- generar predicciones futuras,
- comparar histórico reciente frente a predicción,
- consultar el estado del modelo,
- visualizar una predicción resumida por día,
- revisar la tabla de predicciones,
- descargar resultados en CSV.

#### **Acerca de**
Incluye una breve descripción del propósito general de la aplicación.

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

`App.py` importa el objeto `app` desde `app.main` y ejecuta la aplicación principal.

---

## 📦 Dependencias principales

Este proyecto utiliza principalmente:

- `pandas`
- `matplotlib`
- `plotly`
- `scikit-learn`
- `joblib`
- `shiny`
- `shinywidgets`
- `openpyxl`

Dependencias adicionales del proyecto experimental incluyen librerías como `statsmodels` y `prophet`, utilizadas en la etapa comparativa de modelos.

---

## 📊 Resultados experimentales

Las estrategias evaluadas mostraron que:

- **ARIMA** presenta limitaciones para reproducir adecuadamente la variabilidad observada.
- **Prophet** puede capturar parte de la estructura temporal, pero degrada su desempeño en horizontes largos.
- **Random Forest** mostró el mejor equilibrio entre error, estabilidad y capacidad de ajuste.

Por esta razón, **Random Forest** fue seleccionado como el modelo final integrado en la aplicación.

---

## 📤 Exportación de resultados

La aplicación permite descargar archivos CSV generados a partir de la información filtrada o predicha en las secciones de:

- histórico energético,
- clima,
- predicción.

Estas exportaciones facilitan el análisis posterior de resultados fuera de la interfaz principal.

---

## ⚠️ Limitaciones

- El modelo actual trabaja sobre la señal energética o **generación del sistema fotovoltaico**.
- No predice directamente el **consumo facturable** de la vivienda.
- La calidad de la predicción depende de la consistencia del dataset histórico y de la disponibilidad de variables explicativas.
- Existe un módulo económico auxiliar en el proyecto, pero no forma parte del flujo principal actual de la interfaz.

---

## 🚀 Trabajo futuro

- incorporar indicadores clave de desempeño (KPIs),
- generar reportes automatizados,
- ampliar mecanismos de exportación,
- comparar predicción contra valores observados futuros,
- fortalecer la actualización continua de datos,
- evaluar modelos adicionales de series temporales y aprendizaje profundo,
- seguir mejorando la modularización y visualización de la aplicación.

---

## 📍 Estado del proyecto

Proyecto académico funcional en fase de consolidación, con:

- modelo final entrenado e integrado,
- visualización histórica y climática,
- generación de predicciones,
- gráficas interactivas,
- tablas de consulta,
- exportación CSV por módulo.

---

## 👤 Autor

**Jorge Antonio Carbajal Aguilar**

---

## 🎓 Contexto académico

Proyecto desarrollado en la  
**Universidad Autónoma de Coahuila**  
**Facultad de Sistemas**
