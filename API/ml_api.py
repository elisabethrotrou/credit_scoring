from fastapi import FastAPI, Request, status
from pydantic import BaseModel
import joblib
import json
import logging
import pandas as pd
#import numpy as np
from sklearn import set_config
set_config(transform_output="pandas")
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

# creating the API
api = FastAPI()

@api.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	logging.error(f"{request}: {exc_str}")
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

class ModelInput(BaseModel):
    NAME_CONTRACT_TYPE : object
    NAME_INCOME_TYPE : object
    NAME_EDUCATION_TYPE : object
    NAME_FAMILY_STATUS : object
    NAME_HOUSING_TYPE : object
    WEEKDAY_APPR_PROCESS_START : object
    ORGANIZATION_TYPE : object
    CODE_GENDER : int
    FLAG_OWN_CAR : int
    FLAG_OWN_REALTY : int
    CNT_CHILDREN : int
    AMT_INCOME_TOTAL : float
    AMT_CREDIT : float
    REGION_POPULATION_RELATIVE : float
    DAYS_BIRTH : int
    DAYS_EMPLOYED : int
    DAYS_REGISTRATION : float
    DAYS_ID_PUBLISH : int
    FLAG_MOBIL : int
    FLAG_EMP_PHONE : int
    FLAG_WORK_PHONE : int
    FLAG_CONT_MOBILE : int
    FLAG_PHONE : int
    FLAG_EMAIL : int
    REGION_RATING_CLIENT : int
    REGION_RATING_CLIENT_W_CITY : int
    HOUR_APPR_PROCESS_START : int
    REG_REGION_NOT_LIVE_REGION : int
    REG_REGION_NOT_WORK_REGION : int
    LIVE_REGION_NOT_WORK_REGION : int
    REG_CITY_NOT_LIVE_CITY : int
    REG_CITY_NOT_WORK_CITY : int
    LIVE_CITY_NOT_WORK_CITY : int
    FLAG_DOCUMENT_2 : int
    FLAG_DOCUMENT_3 : int
    FLAG_DOCUMENT_4 : int
    FLAG_DOCUMENT_5 : int
    FLAG_DOCUMENT_6 : int
    FLAG_DOCUMENT_7 : int
    FLAG_DOCUMENT_8 : int
    FLAG_DOCUMENT_9 : int
    FLAG_DOCUMENT_10 : int
    FLAG_DOCUMENT_11 : int
    FLAG_DOCUMENT_12 : int
    FLAG_DOCUMENT_13 : int
    FLAG_DOCUMENT_14 : int
    FLAG_DOCUMENT_15 : int
    FLAG_DOCUMENT_16 : int
    FLAG_DOCUMENT_17 : int
    FLAG_DOCUMENT_18 : int
    FLAG_DOCUMENT_19 : int
    FLAG_DOCUMENT_20 : int
    FLAG_DOCUMENT_21 : int
    DAYS_EMPLOYED_PERC : float
    INCOME_CREDIT_PERC : float

# columns in order
ordered_cols = ["NAME_CONTRACT_TYPE",
"NAME_INCOME_TYPE",
"NAME_EDUCATION_TYPE",
"NAME_FAMILY_STATUS",
"NAME_HOUSING_TYPE",
"WEEKDAY_APPR_PROCESS_START",
"ORGANIZATION_TYPE",
"CODE_GENDER",
"FLAG_OWN_CAR",
"FLAG_OWN_REALTY",
"CNT_CHILDREN",
"AMT_INCOME_TOTAL",
"AMT_CREDIT",
"REGION_POPULATION_RELATIVE",
"DAYS_BIRTH",
"DAYS_EMPLOYED",
"DAYS_REGISTRATION",
"DAYS_ID_PUBLISH",
"FLAG_MOBIL",
"FLAG_EMP_PHONE",
"FLAG_WORK_PHONE",
"FLAG_CONT_MOBILE",
"FLAG_PHONE",
"FLAG_EMAIL",
"REGION_RATING_CLIENT",
"REGION_RATING_CLIENT_W_CITY",
"HOUR_APPR_PROCESS_START",
"REG_REGION_NOT_LIVE_REGION",
"REG_REGION_NOT_WORK_REGION",
"LIVE_REGION_NOT_WORK_REGION",
"REG_CITY_NOT_LIVE_CITY",
"REG_CITY_NOT_WORK_CITY",
"LIVE_CITY_NOT_WORK_CITY",
"FLAG_DOCUMENT_2",
"FLAG_DOCUMENT_3",
"FLAG_DOCUMENT_4",
"FLAG_DOCUMENT_5",
"FLAG_DOCUMENT_6",
"FLAG_DOCUMENT_7",
"FLAG_DOCUMENT_8",
"FLAG_DOCUMENT_9",
"FLAG_DOCUMENT_10",
"FLAG_DOCUMENT_11",
"FLAG_DOCUMENT_12",
"FLAG_DOCUMENT_13",
"FLAG_DOCUMENT_14",
"FLAG_DOCUMENT_15",
"FLAG_DOCUMENT_16",
"FLAG_DOCUMENT_17",
"FLAG_DOCUMENT_18",
"FLAG_DOCUMENT_19",
"FLAG_DOCUMENT_20",
"FLAG_DOCUMENT_21",
"DAYS_EMPLOYED_PERC",
"INCOME_CREDIT_PERC"
]

# loading the saved model
scoring_model = joblib.load('model_scoring.joblib')

# function to pass columns
def model_predict_proba(data_asarray):
    data_asframe =  pd.DataFrame(data_asarray, columns = ordered_cols)
    return scoring_model.predict_proba(data_asframe)

# prediction endpoint
@api.post('/scoring_prediction')

def scoring_pred(input_parameters: ModelInput):
    input_data = input_parameters.json()
    input_dict = json.loads(input_data)

    inp0 = input_dict['NAME_CONTRACT_TYPE']
    inp1 = input_dict['NAME_INCOME_TYPE']
    inp2 = input_dict['NAME_EDUCATION_TYPE']
    inp3 = input_dict['NAME_FAMILY_STATUS']
    inp4 = input_dict['NAME_HOUSING_TYPE']
    inp5 = input_dict['WEEKDAY_APPR_PROCESS_START']
    inp6 = input_dict['ORGANIZATION_TYPE']
    inp7 = input_dict['CODE_GENDER']
    inp8 = input_dict['FLAG_OWN_CAR']
    inp9 = input_dict['FLAG_OWN_REALTY']
    inp10 = input_dict['CNT_CHILDREN']
    inp11 = input_dict['AMT_INCOME_TOTAL']
    inp12 = input_dict['AMT_CREDIT']
    inp13 = input_dict['REGION_POPULATION_RELATIVE']
    inp14 = input_dict['DAYS_BIRTH']
    inp15 = input_dict['DAYS_EMPLOYED']
    inp16 = input_dict['DAYS_REGISTRATION']
    inp17 = input_dict['DAYS_ID_PUBLISH']
    inp18 = input_dict['FLAG_MOBIL']
    inp19 = input_dict['FLAG_EMP_PHONE']
    inp20 = input_dict['FLAG_WORK_PHONE']
    inp21 = input_dict['FLAG_CONT_MOBILE']
    inp22 = input_dict['FLAG_PHONE']
    inp23 = input_dict['FLAG_EMAIL']
    inp24 = input_dict['REGION_RATING_CLIENT']
    inp25 = input_dict['REGION_RATING_CLIENT_W_CITY']
    inp26 = input_dict['HOUR_APPR_PROCESS_START']
    inp27 = input_dict['REG_REGION_NOT_LIVE_REGION']
    inp28 = input_dict['REG_REGION_NOT_WORK_REGION']
    inp29 = input_dict['LIVE_REGION_NOT_WORK_REGION']
    inp30 = input_dict['REG_CITY_NOT_LIVE_CITY']
    inp31 = input_dict['REG_CITY_NOT_WORK_CITY']
    inp32 = input_dict['LIVE_CITY_NOT_WORK_CITY']
    inp33 = input_dict['FLAG_DOCUMENT_2']
    inp34 = input_dict['FLAG_DOCUMENT_3']
    inp35 = input_dict['FLAG_DOCUMENT_4']
    inp36 = input_dict['FLAG_DOCUMENT_5']
    inp37 = input_dict['FLAG_DOCUMENT_6']
    inp38 = input_dict['FLAG_DOCUMENT_7']
    inp39 = input_dict['FLAG_DOCUMENT_8']
    inp40 = input_dict['FLAG_DOCUMENT_9']
    inp41 = input_dict['FLAG_DOCUMENT_10']
    inp42 = input_dict['FLAG_DOCUMENT_11']
    inp43 = input_dict['FLAG_DOCUMENT_12']
    inp44 = input_dict['FLAG_DOCUMENT_13']
    inp45 = input_dict['FLAG_DOCUMENT_14']
    inp46 = input_dict['FLAG_DOCUMENT_15']
    inp47 = input_dict['FLAG_DOCUMENT_16']
    inp48 = input_dict['FLAG_DOCUMENT_17']
    inp49 = input_dict['FLAG_DOCUMENT_18']
    inp50 = input_dict['FLAG_DOCUMENT_19']
    inp51 = input_dict['FLAG_DOCUMENT_20']
    inp52 = input_dict['FLAG_DOCUMENT_21']
    inp53 = input_dict['DAYS_EMPLOYED_PERC']
    inp54 = input_dict['INCOME_CREDIT_PERC']

    input_list = [inp0, inp1, inp2, inp3, inp4, inp5, inp6, inp7, inp8, inp9, 
                  inp10, inp11, inp12, inp13, inp14, inp15, inp16, inp17, inp18, inp19, 
                  inp20, inp21, inp22, inp23, inp24, inp25, inp26, inp27, inp28, inp29, 
                  inp30, inp31, inp32, inp33, inp34, inp35, inp36, inp37, inp38, inp39, 
                  inp40, inp41, inp42, inp43, inp44, inp45, inp46, inp47, inp48, inp49, 
                  inp50, inp51, inp52, inp53, inp54]

    prediction_proba = model_predict_proba([input_list])
    default_risk = jsonable_encoder(prediction_proba[0,1])

    return JSONResponse(content=default_risk)

####################explanation#########################
# loading the saved shape values
#shap_values_frame = joblib.load('shap_sample.joblib')
#shap_values_array = shap_values_frame.to_numpy()

explanation = joblib.load('explanation_test100.joblib')

# explanation endpoint (old version)
#@api.post('/scoring_explanation')
#async def scoring_exp(request: Request):
#    data = await request.json()
#    index = data['item_id']
#    candidate_shap_row = shap_values_array[index]
#    zip_iterator = zip(ordered_cols, candidate_shap_row.tolist())
#    candidate_shap_dict = dict(zip_iterator)
#    candidate_shap_values = jsonable_encoder(candidate_shap_dict)
    
#    return JSONResponse(content=candidate_shap_values)

# explanation endpoint (new version)
@api.post('/scoring_explanation')
async def scoring_exp(request: Request):
    data = await request.json()
    index = data['item_id']
    candidate_shap_info = explanation[index].values[:,1]
    zip_iterator = zip(ordered_cols, candidate_shap_info.tolist())
    candidate_shap_dict = dict(zip_iterator)
    candidate_shap_values = jsonable_encoder(candidate_shap_dict)
    
    return JSONResponse(content=candidate_shap_values)

if __name__ == '__main__':

    api.run(debug=True, host='0.0.0.0', port=8000)