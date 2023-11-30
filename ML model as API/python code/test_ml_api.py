from ml_api import model_predict_proba, scoring_model, shap_values_array
import pytest
import random

class TestClass:        
    def test_risk_proba():
        data_as_array = [["Cash loans","Working","Higher education","Married","House / apartment","TUESDAY","Kindergarten",
                    0,1,0,0,135000.0,568800.0,0.0189,-19241,-2329,-5170,-812,1,1,0,1,0,1,2,2,18,0,0,0,0,0,0,0,1,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.121,0.2373]]
        risk_probas = model_predict_proba(data_as_array)
        assert risk_probas[0,1]>=0 and risk_probas[0,1]<=1

    def test_candidate_row():
        index = random.randint(0,1000)
        row = shap_values_array[index]
        assert row.shape == (55,)