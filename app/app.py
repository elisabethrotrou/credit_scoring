import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import requests # for API calls
import shap
from streamlit_shap import st_shap
import re

st.set_page_config(
    page_title="Client scoring dashboard",
    page_icon=":sleuth_or_spy:",
    layout="wide",
)

# read csv from a URL
@st.cache_data
def get_old_data(cols) -> pd.DataFrame:
    df = pd.read_csv('./application_train.csv', usecols=cols)
    df = df[:10000] #sampling data to fit within limit (maxMessageSize = 300) since no access to config
    return df

def get_new_data() -> pd.DataFrame:
    df = pd.read_csv('./application_test.csv')
    df = df.iloc[:116,:] #sampling data to match with test data passed in shap_values
    df = df[df['CODE_GENDER'] != 'XNA']
    df = df[df['ORGANIZATION_TYPE'] != 'XNA']
    df = df.reset_index(drop=True)
    return df

def prep_data(df, cols):

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    df = df[cols]

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']

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

new_df = get_new_data()
new_df["SK_ID_CURR"] = new_df["SK_ID_CURR"].astype(int)
transformed_new_df = prep_data(new_df, model_features)

# dashboard title
st.title("Client scoring dashboard")

st.markdown("### Application")
# top-level filters / radio buttons
filter, buffer, info1, info2 = st.columns([2,1,2,2])

with filter:
    application_filter = st.selectbox("Select an application from the list below", pd.unique(new_df["SK_ID_CURR"]))
    index_candidate = new_df.index[new_df['SK_ID_CURR'] == application_filter].tolist()

# info from dataframe
age = -int(new_df.loc[new_df["SK_ID_CURR"] == application_filter,"DAYS_BIRTH"]/365)
#gender = new_df.loc[new_df["SK_ID_CURR"] == application_filter]["CODE_GENDER"].iloc[0]
#family = new_df.loc[new_df["SK_ID_CURR"] == application_filter,"NAME_FAMILY_STATUS"].iloc[0]
education = new_df.loc[new_df["SK_ID_CURR"] == application_filter,"NAME_EDUCATION_TYPE"].iloc[0]
region_disc = new_df.loc[new_df["SK_ID_CURR"] == application_filter,"REG_REGION_NOT_LIVE_REGION"].iloc[0]
region_pop = new_df.loc[new_df["SK_ID_CURR"] == application_filter,"REGION_POPULATION_RELATIVE"].iloc[0]
credit = new_df.loc[new_df["SK_ID_CURR"] == application_filter,"AMT_CREDIT"].iloc[0] 
tenure = -int(new_df.loc[new_df["SK_ID_CURR"] == application_filter,"DAYS_EMPLOYED"]/365)
income = new_df.loc[new_df["SK_ID_CURR"] == application_filter,"AMT_INCOME_TOTAL"].iloc[0] 
income_type = new_df.loc[new_df["SK_ID_CURR"] == application_filter,"NAME_INCOME_TYPE"].iloc[0]
realty = new_df.loc[new_df["SK_ID_CURR"] == application_filter,"FLAG_OWN_REALTY"].iloc[0]
car = new_df.loc[new_df["SK_ID_CURR"] == application_filter,"FLAG_OWN_CAR"].iloc[0]

with info1:
        st.write(age, "years old") 
        st.write("education level", education)
        st.write("living in a region of density {:.2f}".format(region_pop))
        if region_disc == 1: st.write("🚨 warning on address discrepancy")
        st.write("already involved with a credit of {:,.0f}€".format(credit))

with info2:
        st.write("employed for past {:.1f}".format(tenure), "years")
        st.write(income_type, "with yearly income of {:,.0f}€".format(income))
        if realty == "Y": st.write("owns their home")
        else: st.write("does not own their home")
        if car == "Y": st.write("owns their car")
        else: st.write("does not own their car")

st.divider()

# single-element container
app_decision = st.empty()

with app_decision.container():
    st.markdown("### Decision & rationale")
    #two columns
    metrics, explanation = st.columns([1,6])

    # predictions from model via API call (via FastAPI)
    url = 'http://api:8000'
    #pred_url = 'http://0.0.0.0:8000/scoring_prediction'
    pred_endpoint = '/scoring_prediction'
    transformed_candidate_data = transformed_new_df.loc[index_candidate]
    
    pred_response = requests.post(url+pred_endpoint, json=transformed_candidate_data.to_dict(orient='records')[0])

    default_proba = float(pred_response.content.decode())
    default_threshold = 0.4

    decision = "NO ❌" if default_proba >= default_threshold else "YES ✅"

    with metrics:
        # fill in those columns with respective metrics or info
        st.metric(
            label="Should we lend?",
            value = decision
        )
        
        st.metric(
            label="What is the default risk?",
            value="{:.0%}".format(default_proba),
            delta="{:.0%}".format(default_proba-default_threshold) + " vs threshold",
            delta_color="inverse"
        )

    with explanation:
        with st.expander("See explanation"):
            st.write("The 2 charts below show the most *impactful* features for predicting a default risk: the first chart describes the general case, when the second one describes how each feature contributes to the default risk prediction for this specific candidate.")
            st.image("./feature_importance.png")
        with st.spinner('SHAP waterfall plot creation in progress...'):
            
            # manual retrieval loading the saved shape values
            #shap_values_frame = joblib.load('shap_sample.joblib')
            #shap_values_array = shap_values_frame.to_numpy()
            #candidate_shap_values = shap_values_array[index_candidate[0]]

            # explanation from model via API call (via FastAPI)
            item = {"item_id": index_candidate[0]}
            #exp_url = 'http://0.0.0.0:8000/scoring_explanation'
            exp_endpoint = '/scoring_explanation'
            exp_response = requests.post(url+exp_endpoint, json=item) 
            candidate_shap_dict = exp_response.json()
            candidate_shap_values_API = np.array(list(candidate_shap_dict.values()))
            candidate_shap_features_API = list(candidate_shap_dict.keys())

            # visualize the candidate's decision explanation
            plot = shap.force_plot(-1.67494564,
                                candidate_shap_values_API,
                                candidate_shap_features_API,
                                link="logit"
                                )
            st_shap(plot, height=120, width=1100)  

st.divider()
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
        ['Age (5yr bucket)', 'Education', 'Income type', 'Employment tenure (2yr bucket)']) 
    
    similars = get_old_data(model_features + ["TARGET"]) #replace with top 20 features in terms of importance + target

    if 'Age (5yr bucket)' in options:
        age_delta = -similars['DAYS_BIRTH']/365 - age 
        similars = similars.loc[abs(age_delta)<5,:]
    else: pass
    if 'Education' in options:
        similars = similars.loc[similars['NAME_EDUCATION_TYPE'] == education,:]
    else: pass
    if 'Income type' in options:
        similars = similars.loc[similars['NAME_INCOME_TYPE'] == income_type,:]
    else: pass
    if 'Employment tenure (2yr bucket)' in options:
        tenure_delta = -similars['DAYS_EMPLOYED']/365 - tenure 
        similars = similars.loc[abs(tenure_delta)<2,:]
    else: pass

    if similars.shape[0] == 0: 
        st.write(":red[The comparison scope is too narrow ; please pick less options above] :arrow_double_up:")
    else: 
        st.write(similars.shape[0], "similar applications and relative distributions of major criteria to review below :arrow_double_down:")
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["income distribution", "credit distribution", "table with values"])
        with tab1:
            # Create histogram
            fig = px.histogram(
                    similars,
                    x='AMT_INCOME_TOTAL',
                    color='TARGET',
                    )
            fig.add_vline(x=income, line_dash = 'dash', line_color = 'firebrick', annotation_text="candidate")
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        with tab2:
            # Create histogram
            fig2 = px.histogram(
                    similars,
                    x='AMT_CREDIT',
                    color='TARGET',
                    )
            fig2.add_vline(x=credit, line_dash = 'dash', line_color = 'firebrick', annotation_text="candidate")
            st.plotly_chart(fig2, theme="streamlit", use_container_width=True)            
        with tab3:
            st.dataframe(similars)
