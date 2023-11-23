from fastapi import FastAPI, Request, status
from pydantic import BaseModel
import joblib
import json
import logging
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

api = FastAPI()

@api.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	logging.error(f"{request}: {exc_str}")
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

class model_input(BaseModel):
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
    NAME_CONTRACT_TYPE : object
    NAME_INCOME_TYPE : object
    NAME_EDUCATION_TYPE : object
    NAME_FAMILY_STATUS : object
    NAME_HOUSING_TYPE : object
    WEEKDAY_APPR_PROCESS_START : object
    ORGANIZATION_TYPE : object
    DAYS_EMPLOYED_PERC : float
    INCOME_CREDIT_PERC : float

# loading the saved model
scoring_model = joblib.load('model_scoring.joblib')

# loading the saved model
#explainer = joblib.load('explainer.joblib')

# transformation nécessaire pour l'importance locale via SHAP
def model_predict_proba(data_asarray):
    data_asframe =  pd.DataFrame(data_asarray)
    return scoring_model.predict_proba(data_asframe)

# creating the API
@api.post('/scoring_prediction')

def scoring_pred(input_parameters: model_input):
    input_data = input_parameters.json()
    input_dict = json.loads(input_data)

    inp0 = input_dict['CODE_GENDER']
    inp1 = input_dict['FLAG_OWN_CAR']
    inp2 = input_dict['FLAG_OWN_REALTY']
    inp3 = input_dict['CNT_CHILDREN']
    inp4 = input_dict['AMT_INCOME_TOTAL']
    inp5 = input_dict['AMT_CREDIT']
    inp6 = input_dict['REGION_POPULATION_RELATIVE']
    inp7 = input_dict['DAYS_BIRTH']
    inp8 = input_dict['DAYS_EMPLOYED']
    inp9 = input_dict['DAYS_REGISTRATION']
    inp10 = input_dict['DAYS_ID_PUBLISH']
    inp11 = input_dict['FLAG_MOBIL']
    inp12 = input_dict['FLAG_EMP_PHONE']
    inp13 = input_dict['FLAG_WORK_PHONE']
    inp14 = input_dict['FLAG_CONT_MOBILE']
    inp15 = input_dict['FLAG_PHONE']
    inp16 = input_dict['FLAG_EMAIL']
    inp17 = input_dict['REGION_RATING_CLIENT']
    inp18 = input_dict['REGION_RATING_CLIENT_W_CITY']
    inp19 = input_dict['HOUR_APPR_PROCESS_START']
    inp20 = input_dict['REG_REGION_NOT_LIVE_REGION']
    inp21 = input_dict['REG_REGION_NOT_WORK_REGION']
    inp22 = input_dict['LIVE_REGION_NOT_WORK_REGION']
    inp23 = input_dict['REG_CITY_NOT_LIVE_CITY']
    inp24 = input_dict['REG_CITY_NOT_WORK_CITY']
    inp25 = input_dict['LIVE_CITY_NOT_WORK_CITY']
    inp26 = input_dict['FLAG_DOCUMENT_2']
    inp27 = input_dict['FLAG_DOCUMENT_3']
    inp28 = input_dict['FLAG_DOCUMENT_4']
    inp29 = input_dict['FLAG_DOCUMENT_5']
    inp30 = input_dict['FLAG_DOCUMENT_6']
    inp31 = input_dict['FLAG_DOCUMENT_7']
    inp32 = input_dict['FLAG_DOCUMENT_8']
    inp33 = input_dict['FLAG_DOCUMENT_9']
    inp34 = input_dict['FLAG_DOCUMENT_10']
    inp35 = input_dict['FLAG_DOCUMENT_11']
    inp36 = input_dict['FLAG_DOCUMENT_12']
    inp37 = input_dict['FLAG_DOCUMENT_13']
    inp38 = input_dict['FLAG_DOCUMENT_14']
    inp39 = input_dict['FLAG_DOCUMENT_15']
    inp40 = input_dict['FLAG_DOCUMENT_16']
    inp41 = input_dict['FLAG_DOCUMENT_17']
    inp42 = input_dict['FLAG_DOCUMENT_18']
    inp43 = input_dict['FLAG_DOCUMENT_19']
    inp44 = input_dict['FLAG_DOCUMENT_20']
    inp45 = input_dict['FLAG_DOCUMENT_21']
    inp46 = input_dict['NAME_CONTRACT_TYPE']
    inp47 = input_dict['NAME_INCOME_TYPE']
    inp48 = input_dict['NAME_EDUCATION_TYPE']
    inp49 = input_dict['NAME_FAMILY_STATUS']
    inp50 = input_dict['WEEKDAY_APPR_PROCESS_START']
    inp51 = input_dict['ORGANIZATION_TYPE']
    inp52 = input_dict['DAYS_EMPLOYED_PERC']
    inp53 = input_dict['INCOME_CREDIT_PERC']
    inp54 = input_dict['NAME_HOUSING_TYPE']


    input_list = [inp0, inp1, inp2, inp3, inp4, inp5, inp6, inp7, inp8, inp9, 
                  inp10, inp11, inp12, inp13, inp14, inp15, inp16, inp17, inp18, inp19, 
                  inp20, inp21, inp22, inp23, inp24, inp25, inp26, inp27, inp28, inp29, 
                  inp30, inp31, inp32, inp33, inp34, inp35, inp36, inp37, inp38, inp39, 
                  inp40, inp41, inp42, inp43, inp44, inp45, inp46, inp47, inp48, inp49, 
                  inp50, inp51, inp52, inp53, inp54]

    prediction = scoring_model.predict([input_list])
    prediction_proba = scoring_model.predict_proba([input_list])
    default_risk = jsonable_encoder(prediction_proba[0,1])

    return JSONResponse(content=default_risk)

    #if prediction[0] == 0:
    #    return "YES ✅"
    #else:
    #    return "NO ❌"