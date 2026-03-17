import unittest
from unittest.mock import patch

import pandas as pd

from app.predict import NotEnoughDataError, choose_model_and_data, filter_scope, predict_family


class ConstantModel:
   def __init__(self, value: float):
      self.value = value

   def predict(self, frame):
      return [self.value]


class PredictModuleTests(unittest.TestCase):

   def test_filter_scope_filters_by_family_member_and_category(self):
      df = pd.DataFrame(
            [
               {
                  "family_id": "fam-1",
                  "family_member_id": "mem-1",
                  "category_id": "cat-1",
                  "year_month": pd.Timestamp("2025-01-01"),
                  "total_expenses": 100,
                  "total_income": 200,
               },
               {
                  "family_id": "fam-1",
                  "family_member_id": "mem-2",
                  "category_id": "cat-1",
                  "year_month": pd.Timestamp("2025-02-01"),
                  "total_expenses": 110,
                  "total_income": 210,
               },
               {
                  "family_id": "fam-1",
                  "family_member_id": "mem-1",
                  "category_id": "cat-2",
                  "year_month": pd.Timestamp("2025-03-01"),
                  "total_expenses": 120,
                  "total_income": 220,
               },
            ]
      )

      filtered = filter_scope(df, "fam-1", "mem-1", "cat-1")

      self.assertEqual(1, len(filtered))
      self.assertEqual("mem-1", filtered.iloc[0]["family_member_id"])
      self.assertEqual("cat-1", filtered.iloc[0]["category_id"])

   def test_filter_scope_raises_when_member_column_is_missing(self):
      df = pd.DataFrame(
            [
               {
                  "family_id": "fam-1",
                  "category_id": "cat-1",
                  "year_month": pd.Timestamp("2025-01-01"),
                  "total_expenses": 100,
                  "total_income": 200,
               }
            ]
      )

      with self.assertRaises(NotEnoughDataError):
         filter_scope(df, "fam-1", "mem-1", None)

   @patch("app.predict.load_history")
   @patch("app.predict.choose_model_and_data")
   def test_predict_family_returns_twelve_months_for_member_scope(self, mock_choose_model_and_data, mock_load_history):
      mock_choose_model_and_data.return_value = (
            ConstantModel(500.0),
            ConstantModel(1500.0),
            "ignored.csv",
      )
      mock_load_history.return_value = pd.DataFrame(
            [
               {
                  "family_id": "fam-1",
                  "family_member_id": "mem-1",
                  "category_id": "cat-1",
                  "year_month": pd.Timestamp(f"2025-{month:02d}-01"),
                  "month": month,
                  "year": 2025,
                  "total_expenses": 100 + month * 10,
                  "total_income": 1000 + month * 20,
               }
               for month in range(1, 9)
            ]
      )

      predictions = predict_family("fam-1", "mem-1", "cat-1")

      self.assertEqual(12, len(predictions))
      self.assertEqual("2025-09", predictions[0]["date"])
      self.assertEqual(1500.0, predictions[0]["predicted_income"])
      self.assertEqual(500.0, predictions[0]["predicted_expenses"])
      self.assertEqual(1000.0, predictions[0]["predicted_balance"])

   @patch("app.predict.joblib.load")
   def test_choose_model_and_data_uses_category_member_suffix(self, mock_joblib_load):
      expense_model = object()
      income_model = object()
      mock_joblib_load.side_effect = [expense_model, income_model]

      selected_expense_model, selected_income_model, dataset = choose_model_and_data("member-1", "category-1")

      self.assertIs(expense_model, selected_expense_model)
      self.assertIs(income_model, selected_income_model)
      self.assertEqual("family-finance-category-member-data.csv", dataset)
      self.assertEqual(2, mock_joblib_load.call_count)
      self.assertIn("predict_category_member_expenses.pkl", str(mock_joblib_load.call_args_list[0].args[0]))
      self.assertIn("predict_category_member_income.pkl", str(mock_joblib_load.call_args_list[1].args[0]))


if __name__ == "__main__":
   unittest.main()
