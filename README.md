# Family Finance Model Service

Versión actual: `1.0.0`

Servicio de predicción online de **Family Finance**. Este microservicio expone el endpoint de forecasting que utiliza el backend para estimar ingresos, gastos y balance durante los próximos 12 meses.

Está desarrollado con **FastAPI** y sirve modelos preentrenados de **scikit-learn** generados por el pipeline offline de entrenamiento.

## Descripción

Este proyecto está separado intencionadamente de la API principal para:

- aislar la inferencia ML del backend de negocio
- cargar artefactos entrenados de forma rápida en tiempo de ejecución
- actualizar modelos de manera independiente al frontend y la API
- mantener una arquitectura de microservicios más limpia

## Qué predice

Según los filtros enviados, el servicio puede generar predicciones a distintos niveles:

- familia
- miembro de la familia
- categoría
- categoría dentro de un miembro de la familia

Para cada punto de predicción devuelve:

- ingreso estimado
- gasto estimado
- balance estimado

## Endpoint

```http
GET /predict?family_id={id}&family_member_id={id?}&category_id={id?}
```

Ejemplo de respuesta:

```json
[
  {
    "date": "2026-04",
    "predicted_income": 2800.0,
    "predicted_expenses": 1950.0,
    "predicted_balance": 850.0
  }
]
```

Si no existe suficiente histórico, el servicio devuelve `400 Bad Request`.

## Cómo funciona

En tiempo de ejecución el servicio:

1. Selecciona el dataset y el par de modelos adecuados según el alcance de la consulta
2. Carga agregados históricos mensuales desde el directorio `data/`
3. Reconstruye features de lags, medias y tendencia
4. Ejecuta una predicción recursiva para los próximos 12 meses
5. Devuelve una respuesta JSON normalizada al backend

## Modelos y datasets

### Datasets

- `data/family-finance-family-data.csv`
- `data/family-finance-member-data.csv`
- `data/family-finance-category-data.csv`
- `data/family-finance-category-member-data.csv`

### Modelos

- `models/predict_family_expenses.pkl`
- `models/predict_family_income.pkl`
- `models/predict_family_member_expenses.pkl`
- `models/predict_family_member_income.pkl`
- `models/predict_category_expenses.pkl`
- `models/predict_category_income.pkl`
- `models/predict_category_member_expenses.pkl`
- `models/predict_category_member_income.pkl`

Estos artefactos son generados por `family-finance-ml-jobs`.

## Stack tecnológico

- Python 3.11
- FastAPI
- Uvicorn
- pandas
- scikit-learn
- joblib

## Ejecución en local

### Requisitos

- Python 3.11
- Modelos entrenados dentro de `models/`
- Datasets CSV dentro de `data/`

### Instalar dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Iniciar la API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

El servicio estará disponible en `http://localhost:8000`.

### Ejecutar tests

```bash
python3 -m unittest discover -s tests
```

Los tests cubren:

- selección de modelos y datasets según el alcance de la predicción
- filtrado por `family_id`, `family_member_id` y `category_id`
- respuesta del endpoint `/predict` en casos correctos y de error

## Docker

Este servicio incluye un Dockerfile preparado para producción y está integrado en el Docker Compose del repositorio.

Desde la raíz del proyecto:

```bash
docker compose up --build
```

## Notebooks

El directorio `notebooks/` contiene el trabajo exploratorio y de entrenamiento utilizado para definir el enfoque actual:

- notebooks de extracción de datasets
- notebooks de entrenamiento para cada nivel de predicción

Esto aporta valor en portfolio porque enseña tanto la parte de experimentación como la de servicio productivo.

## Proyectos relacionados

- `family-finance-api`: API segura que consume este servicio
- `family-finance-ml-jobs`: generación batch de datasets y modelos entrenados
- `family-finance-web`: frontend que muestra las predicciones al usuario
