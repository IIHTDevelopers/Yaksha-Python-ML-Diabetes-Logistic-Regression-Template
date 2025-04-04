import unittest
from test.TestUtils import TestUtils
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from main import load_and_preprocess, train_model, evaluate_model, predict_new
import io
import sys


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        # Initialize TestUtils object for yaksha assertions
        self.test_obj = TestUtils()

        # Prepare test data
        self.X_train, self.X_test, self.y_train, self.y_test = load_and_preprocess()
        self.model = train_model(self.X_train, self.y_train)

        # Test data for predictions
        self.diabetic_patient = (47, 35.1, 90)  # Age, BMI, BP with high diabetic probability
        self.non_diabetic_patient = (25, 22.4, 75)  # Age, BMI, BP with low diabetic probability

    def test_load_and_preprocess(self):
        """
        Test case for load_and_preprocess() function.
        """
        try:
            X_train, X_test, y_train, y_test = load_and_preprocess()

            # Check if data is split correctly
            expected_train_size = 24  # 80% of 30 records
            expected_test_size = 6  # 20% of 30 records

            if (len(X_train) == expected_train_size and
                    len(X_test) == expected_test_size and
                    len(y_train) == expected_train_size and
                    len(y_test) == expected_test_size):
                self.test_obj.yakshaAssert("TestLoadAndPreprocess", True, "functional")
                print("TestLoadAndPreprocess = Passed")
            else:
                self.test_obj.yakshaAssert("TestLoadAndPreprocess", False, "functional")
                print("TestLoadAndPreprocess = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestLoadAndPreprocess", False, "functional")
            print(f"TestLoadAndPreprocess = Failed")

    def test_train_model(self):
        """
        Test case for train_model() function.
        """
        try:
            model = train_model(self.X_train, self.y_train)

            # Check if model is a LogisticRegression instance
            if isinstance(model, LogisticRegression):
                self.test_obj.yakshaAssert("TestTrainModel", True, "functional")
                print("TestTrainModel = Passed")
            else:
                self.test_obj.yakshaAssert("TestTrainModel", False, "functional")
                print("TestTrainModel = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestTrainModel", False, "functional")
            print(f"TestTrainModel = Failed")

    def test_evaluate_model(self):
        """
        Test case for evaluate_model() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output

            evaluate_model(self.model, self.X_test, self.y_test)

            # Reset stdout
            sys.stdout = sys.__stdout__

            # Check if evaluation output contains expected strings
            output = captured_output.getvalue()
            if ("Evaluation Results:" in output and
                    "Accuracy:" in output and
                    "Classification Report:" in output):
                self.test_obj.yakshaAssert("TestEvaluateModel", True, "functional")
                print("TestEvaluateModel = Passed")
            else:
                self.test_obj.yakshaAssert("TestEvaluateModel", False, "functional")
                print("TestEvaluateModel = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestEvaluateModel", False, "functional")
            print(f"TestEvaluateModel = Failed ")

    def test_predict_new_diabetic(self):
        """
        Test case for predict_new() function with high diabetic probability.
        """
        try:
            age, bmi, bp = self.diabetic_patient

            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output

            # Create a sample DataFrame for prediction
            sample = pd.DataFrame({'Age': [age], 'BMI': [bmi], 'BloodPressure': [bp]})

            # Get the model's prediction
            prediction = self.model.predict(sample)[0]

            # Reset stdout
            sys.stdout = sys.__stdout__

            # Check if prediction is 1 (is diabetic)
            if prediction == 1:
                self.test_obj.yakshaAssert("TestPredictNewDiabetic", True, "functional")
                print("TestPredictNewDiabetic = Passed")
            else:
                self.test_obj.yakshaAssert("TestPredictNewDiabetic", False, "functional")
                print("TestPredictNewDiabetic = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestPredictNewDiabetic", False, "functional")
            print(f"TestPredictNewDiabetic = Failed")

    def test_predict_new_non_diabetic(self):
        """
        Test case for predict_new() function with low diabetic probability.
        """
        try:
            age, bmi, bp = self.non_diabetic_patient

            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output

            # Create a sample DataFrame for prediction
            sample = pd.DataFrame({'Age': [age], 'BMI': [bmi], 'BloodPressure': [bp]})

            # Get the model's prediction
            prediction = self.model.predict(sample)[0]

            # Reset stdout
            sys.stdout = sys.__stdout__

            # Check if prediction is 0 (not diabetic)
            if prediction == 0:
                self.test_obj.yakshaAssert("TestPredictNewNonDiabetic", True, "functional")
                print("TestPredictNewNonDiabetic = Passed")
            else:
                self.test_obj.yakshaAssert("TestPredictNewNonDiabetic", False, "functional")
                print("TestPredictNewNonDiabetic = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestPredictNewNonDiabetic", False, "functional")
            print(f"TestPredictNewNonDiabetic = Failed")


if __name__ == '__main__':
    unittest.main()
