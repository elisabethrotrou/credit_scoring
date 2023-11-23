import time
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import requests # for API calls
import shap
import streamlit.components.v1 as components
import json
import re
import joblib

st.set_page_config(
    page_title="Client scoring dashboard",
    page_icon=":sleuth_or_spy:",
    layout="wide",
)

# read csv from a github repo
#dataset_url = "https://raw.githubusercontent.com/Lexie88rus/bank-marketing-analysis/master/bank.csv"

# read csv from a URL
@st.cache_data


def get_old_data(cols) -> pd.DataFrame:
    df = pd.read_csv('./input/application_train.csv', usecols=cols)
    df = df[:100000] #sampling data to fit within limit (maxMessageSize = 300) since no access to config
    return df

def get_new_data() -> pd.DataFrame:
    return pd.read_csv('./input/application_test.csv')

def prep_data(df, cols):
    df = df[df['CODE_GENDER'] != 'XNA']
    df = df[df['ORGANIZATION_TYPE'] != 'XNA']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
   
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    df = df[cols]
    return df

model_features = ['AMT_CREDIT','AMT_INCOME_TOTAL','CNT_CHILDREN','CODE_GENDER','DAYS_BIRTH','DAYS_EMPLOYED',
         'DAYS_ID_PUBLISH','DAYS_REGISTRATION','FLAG_CONT_MOBILE','FLAG_DOCUMENT_10',
         'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
         'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_2',
         'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
         'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_EMAIL', 'FLAG_EMP_PHONE',
         'FLAG_MOBIL', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_PHONE', 'FLAG_WORK_PHONE', 'HOUR_APPR_PROCESS_START',
         'LIVE_CITY_NOT_WORK_CITY', 'LIVE_REGION_NOT_WORK_REGION', 'NAME_CONTRACT_TYPE',
         'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE', 'ORGANIZATION_TYPE',
         'REGION_POPULATION_RELATIVE', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'REG_CITY_NOT_LIVE_CITY',
         'REG_CITY_NOT_WORK_CITY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION','WEEKDAY_APPR_PROCESS_START']

old_df = get_old_data(model_features)
new_df = get_new_data()
transformed_new_df = prep_data(new_df, model_features)

# dashboard title
st.title("Client scoring dashboard")

# top-level filters / radio buttons

#accessibility, buffer = st.columns(2)
#with accessibility:
#    st.checkbox("BIG FONT")
#    text_size = st.slider('How big do you need the text to be?', 1, 200)

filter, buffer, score, risk = st.columns([1,1,1,1])

with filter:
    application_filter = st.selectbox("Select an application from the list below", pd.unique(new_df["SK_ID_CURR"]))
    index_candidate = new_df.index[new_df['SK_ID_CURR'] == application_filter].tolist()


# predictions from model via API call (via FastAPI)

#url = 'http://127.0.0.1:8000/scoring_prediction'

#transformed_candidate_data = transformed_new_df.loc[index_candidate]

#response = requests.post(url, json=transformed_candidate_data.to_dict(orient='records')[0])

#default_proba = response.content.decode()
default_proba = 0.2 #float(default_proba)

default_threshold = 0.3

decision = "NO ❌" if default_proba >= default_threshold else "YES ✅"

# info from dataframe
age = -int(new_df.loc[new_df["SK_ID_CURR"] == application_filter,"DAYS_BIRTH"]/365)
gender = new_df.loc[new_df["SK_ID_CURR"] == application_filter]["CODE_GENDER"].iloc[0]
family = new_df.loc[new_df["SK_ID_CURR"] == application_filter,"NAME_FAMILY_STATUS"].iloc[0]
education = new_df.loc[new_df["SK_ID_CURR"] == application_filter,"NAME_EDUCATION_TYPE"].iloc[0]
income = new_df.loc[new_df["SK_ID_CURR"] == application_filter,"NAME_INCOME_TYPE"].iloc[0]
realty = new_df.loc[new_df["SK_ID_CURR"] == application_filter,"FLAG_OWN_REALTY"].iloc[0]
car = new_df.loc[new_df["SK_ID_CURR"] == application_filter,"FLAG_OWN_CAR"].iloc[0]

with score:
    # fill in those columns with respective metrics or info
    score.metric(
        label="Lending decision",
        value = decision
        #delta=round(avg_age) - 10,
    )

with risk:        
    risk.metric(
        label="Default risk level",
        value="{:.0%}".format(default_proba),
        delta="{:.0%}".format(default_proba-default_threshold) + " vs threshold",
        delta_color="inverse"
    )



# single-element container
application = st.empty()

with application.container():

    #two columns
    info, shap = st.columns([2,2])

    with info:

        st.write("### Application details")
        st.write(age, "years old", 
                family, gender, "with", education)
        #st.write("*Applicant* is a", )
        #st.write("*Applicant* marital status is", )
        #st.write("*Applicant* education level is", education)
        st.write("income earned", income)
        #st.write("*Applicant* income type is", income)
        if realty == "Y": st.write("owns their home")
        else: st.write("does not own their home")
        if car == "Y": st.write("owns their car")
        else: st.write("does not own their car")

    with shap:
        st.markdown("### Decision explanation")

        # function enabling shap plot in html
        def st_shap(plot, height=None):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=height)

        # visualize the candidate's decision explanation
        #st_shap(shap.force_plot(explainer.expected_value[0], np.array(shap_values[0]), transformed_candidate_data,
        #                        link='logit', out_names='risque de défaut'), height=400)
            

# single-element container
past_applications = st.empty()

with past_applications.container():

    st.markdown("### Similar past applications")
        
    #changing multi-select labels color from red which looks alarming
    st.markdown(
    """
    <style>
    span[data-baseweb="tag"] {
    background-color: blue !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
        
    options = st.multiselect(
        'Select a comparison scope from the caracteristics below (you can pick more than one)',
        ['Age (5yr bucket)', 'Gender', 'Family', 'Education', 'Income type']) 
    
    similars = old_df

    if 'Age (5yr bucket)' in options:
        age_delta = -similars['DAYS_BIRTH']/365 - age 
        similars = similars.loc[abs(age_delta)<5,:]
    else: pass
    if 'Gender' in options:
        similars = similars.loc[similars['CODE_GENDER'] == gender,:]
    else: pass
    if 'Family' in options:
        similars = similars.loc[similars['NAME_FAMILY_STATUS'] == family,:]
    else: pass
    if 'Education' in options:
        similars = similars.loc[similars['NAME_EDUCATION_TYPE'] == education,:]
    else: pass
    if 'Income type' in options:
        similars = similars.loc[similars['NAME_INCOME_TYPE'] == income,:]
    else: pass

    if options:
        if similars.shape[0] != 0: 
            st.write(similars.shape[0], "similar applications to review below :arrow_double_down:")
        else: 
            st.write(":red[The comparison scope is too narrow ; please pick less options above] :arrow_double_up:")
    else:pass
    
    if not options:
        pass
    else:
        if similars.shape[0] == 0: 
            pass
        else:
            #similars = similars.set_index('SK_ID_CURR', drop=True, inplace=True) 
            st.dataframe(similars)
    #time.sleep(1)