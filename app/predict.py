import pandas as pd
import joblib
from dateutil.relativedelta import relativedelta
from pathlib import Path

class NotEnoughDataError(Exception):
   pass

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
HISTORY_WINDOW = 6
MIN_HISTORY_MONTHS = 7


def load_history(file_data: str) -> pd.DataFrame:
   dataset_path = DATA_DIR / file_data
   if not dataset_path.exists():
      raise FileNotFoundError(f"Dataset not found: {dataset_path}")

   df = pd.read_csv(dataset_path)
   df["year_month"] = pd.to_datetime(df["year_month"])
   df["month"] = df["year_month"].dt.month
   df["year"] = df["year_month"].dt.year
   return df


def filter_scope(
   df: pd.DataFrame,
   family_id: str | None,
   family_member_id: str | None,
   category_id: str | None,
) -> pd.DataFrame:
   filtered = df.copy()

   if family_id:
      filtered = filtered[filtered["family_id"] == family_id]

   if family_member_id:
      if "family_member_id" not in filtered.columns:
         raise NotEnoughDataError("Member-level prediction is not available for this dataset")
      filtered = filtered[filtered["family_member_id"] == family_member_id]

   if category_id:
      if "category_id" not in filtered.columns:
         raise NotEnoughDataError("Category-level prediction is not available for this dataset")
      filtered = filtered[filtered["category_id"] == category_id]

   return filtered.sort_values("year_month").reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
   prepared = df.copy()
   prepared["exp_lag1"] = prepared["total_expenses"].shift(1)
   prepared["exp_lag2"] = prepared["total_expenses"].shift(2)
   prepared["exp_lag3"] = prepared["total_expenses"].shift(3)
   prepared["exp_lag6"] = prepared["total_expenses"].shift(HISTORY_WINDOW)

   prepared["inc_lag1"] = prepared["total_income"].shift(1)
   prepared["inc_lag2"] = prepared["total_income"].shift(2)
   prepared["inc_lag3"] = prepared["total_income"].shift(3)
   prepared["inc_lag6"] = prepared["total_income"].shift(HISTORY_WINDOW)

   prepared["exp_avg3"] = prepared[["exp_lag1", "exp_lag2", "exp_lag3"]].mean(axis=1)
   prepared["inc_avg3"] = prepared[["inc_lag1", "inc_lag2", "inc_lag3"]].mean(axis=1)

   prepared["exp_avg6"] = (
      prepared["total_expenses"]
      .shift(1)
      .rolling(HISTORY_WINDOW)
      .mean()
   )
   prepared["inc_avg6"] = (
      prepared["total_income"]
      .shift(1)
      .rolling(HISTORY_WINDOW)
      .mean()
   )

   prepared["exp_trend"] = prepared["exp_lag1"] - prepared["exp_lag3"]
   prepared["inc_trend"] = prepared["inc_lag1"] - prepared["inc_lag3"]

   return prepared.dropna().reset_index(drop=True)

# Función de predicción
def predict_family(family_id: str, family_member_id=None, category_id=None):
   # Elegimos el modelo
   model_expenses, model_income, file_data = choose_model_and_data(family_member_id, category_id)

   scoped_history = filter_scope(load_history(file_data), family_id, family_member_id, category_id)
   if len(scoped_history) < MIN_HISTORY_MONTHS:
      raise NotEnoughDataError("Not enough data")

   feature_history = add_features(scoped_history)
   if feature_history.empty:
      raise NotEnoughDataError("Not enough data")

   history_window = scoped_history.iloc[-HISTORY_WINDOW:].copy().reset_index(drop=True)
   start = scoped_history.iloc[-1]["year_month"]

   future_expenses = []
   future_income = []
   future_balance = []

   for i in range(12):
      month = (int(history_window.iloc[-1]["month"]) % 12) + 1

      exp_lag1 = history_window.iloc[-1]["total_expenses"]
      exp_lag2 = history_window.iloc[-2]["total_expenses"]
      exp_lag3 = history_window.iloc[-3]["total_expenses"]
      exp_lag6 = history_window.iloc[-6]["total_expenses"]

      inc_lag1 = history_window.iloc[-1]["total_income"]
      inc_lag2 = history_window.iloc[-2]["total_income"]
      inc_lag3 = history_window.iloc[-3]["total_income"]
      inc_lag6 = history_window.iloc[-6]["total_income"]

      exp_avg3 = (exp_lag1 + exp_lag2 + exp_lag3) / 3
      inc_avg3 = (inc_lag1 + inc_lag2 + inc_lag3) / 3

      exp_avg6 = history_window["total_expenses"].mean()
      inc_avg6 = history_window["total_income"].mean()

      exp_trend = exp_lag1 - exp_lag3
      inc_trend = inc_lag1 - inc_lag3

      X_future = pd.DataFrame([{
         "month": month,
         "exp_lag1": exp_lag1,
         "exp_lag2": exp_lag2,
         "exp_lag3": exp_lag3,
         "exp_lag6": exp_lag6,
         "exp_avg3": exp_avg3,
         "exp_avg6": exp_avg6,
         "exp_trend": exp_trend,
         "inc_lag1": inc_lag1,
         "inc_lag2": inc_lag2,
         "inc_lag3": inc_lag3,
         "inc_lag6": inc_lag6,
         "inc_avg3": inc_avg3,
         "inc_avg6": inc_avg6,
         "inc_trend": inc_trend
      }])

      pred_exp = model_expenses.predict(X_future)[0]
      pred_inc = model_income.predict(X_future)[0]

      balance = pred_inc - pred_exp

      future_expenses.append(pred_exp)
      future_income.append(pred_inc)
      future_balance.append(balance)

      new_row = {
         "total_expenses": pred_exp,
         "total_income": pred_inc,
         "month": month
      }

      history_window = pd.concat(
         [history_window, pd.DataFrame([new_row])],
         ignore_index=True
      ).tail(HISTORY_WINDOW).reset_index(drop=True)

   dates = [
      start + relativedelta(months=i)
      for i in range(1, 13)
   ]

   predictions_df = pd.DataFrame({
      "date": [d.strftime("%Y-%m") for d in dates],
      "predicted_income": future_income,
      "predicted_expenses": future_expenses,
      "predicted_balance": future_balance
   })

   return predictions_df.to_dict(orient="records")

# Elegimos el modelo y el dataset
def choose_model_and_data(family_member_id=None, category_id=None):
   if family_member_id is not None and category_id is not None:
      suffix = "category_member"
      file_data = "family-finance-category-member-data.csv"
      
   elif family_member_id is not None:
      suffix = "family_member"
      file_data = "family-finance-member-data.csv"
      
   elif category_id is not None:
      suffix = "category"
      file_data = "family-finance-category-data.csv"
   else:
      suffix = "family"
      file_data = "family-finance-family-data.csv"

   model_expenses = joblib.load(MODELS_DIR / f"predict_{suffix}_expenses.pkl")
   model_income = joblib.load(MODELS_DIR / f"predict_{suffix}_income.pkl")

   return model_expenses, model_income, file_data
