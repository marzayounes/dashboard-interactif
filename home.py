# ======================== | Imports | ========================

# imports
from email import contentmanager
from pyexpat import model
from urllib import response
from matplotlib.font_manager import json_load
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from zipfile import ZipFile
import joblib
import json
import requests

# Preprocessing, Imputing, Upsampling, Model Selection, Model Evaluation
import sklearn
from sklearn.impute import SimpleImputer
#from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings

from visions import URL
from yaml import load
warnings.filterwarnings("ignore")
import shap
import streamlit.components.v1 as components


# # URL parent du serveur Flask
FLASK_URL = "http://127.0.0.1:8500/"

# ======================== | Data Import | ========================

# ======================== | Page title & sidebar | ========================

st.markdown("# Home page \U0001F3E6")
st.sidebar.markdown("# Home page \U0001F3E6")

# ======================== | Interactions, API calls and decode | ========================

# API calls | GET data (used to select customer idx)
@st.cache_data
def load_data():
    url_data = FLASK_URL + "load_data/"
    response = requests.get(url_data)
    content = json.loads(response.content.decode('utf-8'))
    dict_data = content["data"]
    data = pd.DataFrame.from_dict(eval(dict_data), orient='columns')
    return data
data = load_data()

# Select Customer number SK_ID_CURR in data
idx = st.sidebar.selectbox(
    "Select Credit File", 
    data.SK_ID_CURR, key ="idx")
# Initialization of idx
if 'idx' not in st.session_state:
    st.session_state['idx'] = idx

# GET predict : prediction / prob_predict / ID_to_predict : 
url_predict_client = FLASK_URL + "predict/" + str(idx)
response = requests.get(url_predict_client)
content = json.loads(response.content.decode('utf-8'))
prediction = content["prediction"]
decision = content["decision"]
prob_predict = content["prob_predict"]
dict_ID_to_predict = content["ID_to_predict"]
ID_to_predict = pd.DataFrame.from_dict(eval(dict_ID_to_predict), orient='columns')

# GET top_10
url_top_10 = FLASK_URL + "load_top_10/"
response = requests.get(url_top_10)
content = json.loads(response.content.decode('utf-8'))
top_10 = content["top_10"]

# GET best_thresh
url_best_thresh = FLASK_URL + "load_best_thresh/"
response = requests.get(url_best_thresh)
content = json.loads(response.content.decode('utf-8'))
best_thresh = content["best_thresh"]

# GET X_test and cache it (heavy)
@st.cache_data
def load_X_test():
    url_X_test = FLASK_URL + "load_X_test/"
    response = requests.get(url_X_test)
    content = json.loads(response.content.decode('utf-8'))
    dict_X_test = content["X_test"]
    X_test = pd.DataFrame.from_dict(eval(dict_X_test), orient='columns')
    return X_test
X_test = load_X_test()


# ======================== | Interactions, API calls and decode | ========================

#### INTERACTIONS IN THE STREAMLIT SESSION ####

st.write(f"Customer number : {str(idx)}\n | Credit is " + decision)
st.write("Code_Gender : 1 = Female | 0 = Male")
st.write(ID_to_predict)


# warning message asking to select an action
def customer():
    customer_txt = "Please select an action on the sidebar."

    # checkbox to display customer txt 1 or 2:
    check1 = st.sidebar.checkbox('Show Customer Jauge', key = "check_1")
    check2 = st.sidebar.checkbox('Show Customer Details Top 10', key = "check_2")
    check = check1 + check2   
    if check == 0:
        return check1, check2, st.write(customer_txt)
    else:
        return check1, check2, st.write("")

check1, check2, text = customer()


# displays a jauge with credit analysis result for selected customer
# @st.cache(allow_output_mutation=True, show_spinner=True, suppress_st_warning=True)
def jauge():
    if prob_predict < best_thresh:
        title_auto = {'text':"<b>No probability of default detected</b><br>\
    <span style='color: forestgreen; font-size:0.9em'>Credit<br><b>Granted</b></span>", \
                    'font': {'color': 'forestgreen', 'size': 15}}
    else:
        title_auto = {'text':"<b>Probability of default detected</b><br>\
    <span style='color: crimson; font-size:0.9em'>Credit<br><b>Not granted</b></span>", \
                    'font': {'color': 'crimson', 'size': 15}}


    fig2 = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = prob_predict,
        mode = "gauge+number+delta",
        title = title_auto,
        delta = {'reference': best_thresh},
        gauge = {'axis': {'range': [None, 1]},
                'bgcolor': "crimson",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps' : [
                    {'range': [0, best_thresh], 'color': "forestgreen"},
                    {'range': [best_thresh, 0.5], 'color': "crimson"}],
                'threshold' : {'line': {'color': "crimson", 'width': 2}, 'thickness': 1, 'value': best_thresh},
                'bar': {'color': "palegoldenrod"}}))

    if prob_predict < best_thresh:
        fig2.update_layout(paper_bgcolor = "honeydew", font = {'color': "darkgreen", 'family': "Arial"})
    else:
        fig2.update_layout(paper_bgcolor = "lavenderblush", font = {'color': "crimson", 'family': "Arial"})

    st.write(fig2)

# checkbox to display jauge:
if check1:
    jauge()



# @st.cache(allow_output_mutation=True, show_spinner=True, suppress_st_warning=True)
def customer_details():
    # utiliser graph objects avec une boucle sur top_10
    # pour montrer uniquement des données chiffrées

    sel = top_10
    fig1 = go.Figure()

    for i, c in enumerate(sel,1):
        chaine = "Val / Var Mean :<br>" + c

        if ((i == 1) | (i == 2)):
            row = 0
            column = 1 - i%2
        elif i % 2 != 0:
            row = int(i/2)
            column = 0
        else:
            row = int((i-1)/2)
            column = 1
        
        fig1.add_trace(go.Indicator(
            mode = "number+delta",
            value = ID_to_predict[c].iloc[0],
            delta = {'reference': np.mean(X_test[c]),
                    'valueformat': '.0f',
                    'increasing': {'color': 'green'},
                    'decreasing': {'color': 'red'}},
            title = chaine,
            domain = {'row': row, 'column': column}))

    fig1.update_layout(
        grid = {'rows': 5, 'columns': 2, 'pattern': "independent", 'xgap' : 0.5, 'ygap' : 0.6})

    fig1.update_layout(
        autosize=False,
        width=800,
        height=700,)

    plt.tight_layout()

    st.write(fig1)

# checkbox to display Top 10:
if check2:
    customer_details()