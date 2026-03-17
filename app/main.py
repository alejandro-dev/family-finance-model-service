from fastapi import FastAPI, HTTPException

from app.predict import NotEnoughDataError, predict_family
from app.schemas import PredictionItem

import traceback

app = FastAPI()

@app.get("/predict", response_model=list[PredictionItem])
def predict(family_id: str | None = None, family_member_id: str | None = None, category_id: str | None = None):
   try:
      # Realizamos la predicción
      return predict_family(family_id, family_member_id, category_id)
   
   except NotEnoughDataError as ex:
      # Si no hay suficientes datos, devolvemos un error 400
      raise HTTPException(status_code=400, detail=str(ex)) from ex
   
   except HTTPException:
      # En caso de cualquier otro error, devolvemos un error 500
      raise
   
   except Exception as ex:
      traceback.print_exc()
      print(str(ex))
      # Si hay un error en la respuesta del servidor de predicción, devolvemos un error 503
      raise HTTPException(status_code=503, detail="Prediction service failed") from ex
