from pydantic import BaseModel

# Definimos el esquema de un elemento de predicción
class PredictionItem(BaseModel):
    date: str
    predicted_income: float
    predicted_expenses: float
    predicted_balance: float
