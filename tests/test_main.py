import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.predict import NotEnoughDataError


class MainApiTests(unittest.TestCase):

   def setUp(self):
      self.client = TestClient(app)

   @patch("app.main.predict_family")
   def test_predict_endpoint_returns_predictions(self, mock_predict_family):
      mock_predict_family.return_value = [
            {
               "date": "2025-09",
               "predicted_income": 1500.0,
               "predicted_expenses": 500.0,
               "predicted_balance": 1000.0,
            }
      ]

      response = self.client.get("/predict?family_id=fam-1&family_member_id=mem-1")

      self.assertEqual(200, response.status_code)
      self.assertEqual("2025-09", response.json()[0]["date"])

   @patch("app.main.predict_family")
   def test_predict_endpoint_returns_bad_request_for_not_enough_data(self, mock_predict_family):
      mock_predict_family.side_effect = NotEnoughDataError("Not enough data")

      response = self.client.get("/predict?family_id=fam-1")

      self.assertEqual(400, response.status_code)
      self.assertEqual("Not enough data", response.json()["detail"])

   @patch("app.main.predict_family")
   def test_predict_endpoint_returns_service_unavailable_for_unexpected_errors(self, mock_predict_family):
      mock_predict_family.side_effect = RuntimeError("boom")

      response = self.client.get("/predict?family_id=fam-1")

      self.assertEqual(503, response.status_code)
      self.assertEqual("Prediction service failed", response.json()["detail"])


if __name__ == "__main__":
   unittest.main()
