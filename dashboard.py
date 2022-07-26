import streamlit as st
from PIL import Image
import requests as re
import json
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from functions import *

# local address server
#url = 'http://127.0.0.1:5000/api/'
# Online address server
url = 'https://loan-prediction-oc7.herokuapp.com/api/'
st.title('Loan Prediction Dashboard')
st.markdown("Creator : **_Sabrine OUANNES_**")
st.markdown("Customer : **_Prêt à dépenser_**")
logo = Image.open('assets/logo.JPG')
st.sidebar.image(logo, width=250)
customer_id = st.sidebar.text_input("Customer_ID")
customer = st.sidebar.checkbox("Customer information")
model = st.sidebar.checkbox("Model information")
comparison = st.sidebar.checkbox("Customers comparison")

if customer:
    req = {"SK_ID_CURR": int(customer_id)}
    res = re.post(url + 'predict', json=req)  # to call api
    content = json.loads(res.content)
    if content['statut_code'] == "0":
        st.error(content["erreur"])

    else:
        with st.spinner('Wait for it...'):
            time.sleep(5)
        st.success('Done!')
        st.write("Customer id : " + str(customer_id))
        proba = "YES", "No"
        data = [float(content['proba_yes']), float(content['proba_no'])]
        fig, ax = plt.subplots()
        ax.pie(data, labels=proba, autopct='%1.2f%%', shadow=True, colors=['green', 'red'])
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

        if content['prediction'] == '0':
            st.markdown(f'<h1 style="color:green;font-size:24px;">{"Decision : Loan granted"}</h1>',
                        unsafe_allow_html=True)
            happy = Image.open('assets/happy_emoji.jpg')
            st.image(happy, width=20)

        else:
            # st.write('Decision : Loan refused')
            st.markdown(f'<h1 style="color:red;font-size:24px;">{"Decision : Loan refused"}</h1>',
                        unsafe_allow_html=True)
            sad = Image.open('assets/sad_emoji.png')
            st.image(sad, width=50)
        explanation_lime = st.checkbox('Show explanation lime')
        if explanation_lime:
            lime_df = pd.DataFrame(content["lime_explanation"], columns=['Feature', 'Value_lime'])
            # To define colors
            for i in range(35):
                # Colour of bar chart is set to green if the lime_value
                # is < 0 and red otherwise
                lime_df['colors'] = ['green' if x < 0 else 'red' for x in lime_df['Value_lime']]
            # Draw plot
            fig, ax = plt.subplots(figsize=(10, 10))
            # Plotting the horizontal lines
            ax.hlines(y=lime_df['Feature'], xmin=0, xmax=lime_df['Value_lime'],
                      color=lime_df.colors, alpha=0.4, linewidth=5)
            st.pyplot(fig)

            customer_info = st.checkbox("Show customer information's")
            if customer_info:
                res = re.get(url + 'get/' + str(customer_id))
                content = json.loads(res.content)
                cust_df = pd.DataFrame(data=content['customer_data'], columns=content['Features'])
                Feat_to_show = st.multiselect('Which feature do you want to show', content['Features'])
                if Feat_to_show:
                    st.write(cust_df[Feat_to_show].T)
                else:
                    st.table(cust_df.T)



elif model:
    st.write('ML model : LGBMClassifier')
    st.write('In order to predict that a customer repays his credit or not, we used LGBMClassifier\n'
             'as a machine learning model to classify customers.')
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "70 %")
    col2.metric("ROC_AUC", "78 %")
    col3.metric("F5_score", "63 %")
    feat_importance = st.checkbox("Show features importance of model", value=True)
    if feat_importance:
        res = re.get(url + 'get/importance')
        content = json.loads(res.content)
        feature_imp = pd.DataFrame({'Feature': content['features'],
                                    'Value': content['Values']})

        feature_imp["Value"] = pd.to_numeric(feature_imp["Value"])
        feature_imp = feature_imp.sort_values(by=['Value'], ascending=False)
        num_feat = st.slider("Select a number of the most important features", 1, 20, 1)
        st.write("You have selected the ", num_feat, "features most important for the model")
        fig, ax = plt.subplots()
        bar_data = feature_imp.iloc[:num_feat]
        sns.barplot(x="Value", y="Feature", data=bar_data)
        st.pyplot(fig)
        feat_description = st.checkbox("Show features description")
        if feat_description:
            feat_desc = pd.read_csv('assets/features_description.csv', on_bad_lines='skip', sep=';')
            feat_desc = pd.DataFrame(data=feat_desc)
            st.write(feat_desc)




elif comparison:
    res = re.get(url + 'get/' + str(customer_id))
    content = json.loads(res.content)
    all_features = content["Features"]
    all_data = pd.DataFrame(np.array(list(content['other_customers']), dtype=float), columns=all_features)
    client = pd.DataFrame(np.array(list(content['customer_data']), dtype=float), columns=all_features)
    scaler = StandardScaler()
    data_to_scale = all_data.drop(columns=['TARGET'], axis=1)
    data_to_scale = pd.DataFrame(data=scaler.fit_transform(data_to_scale), columns=data_to_scale.columns)
    client_without_target = client.drop(columns=['TARGET'], axis=1)
    client_without_target = pd.DataFrame(data=scaler.transform(client_without_target),
                                         columns=data_to_scale.columns)
    all_data = pd.concat([data_to_scale, all_data['TARGET']], axis=1)
    all_data['TARGET'].replace({0.0: 'REPAID', 1.0: 'NOT REPAID'}, inplace=True)
    client = pd.concat([client_without_target, client['TARGET']], axis=1)
    data_melt_all = all_data.melt(id_vars=['TARGET'])
    dist_columns = all_data.drop(columns=['TARGET'], axis=1).columns

    with st.spinner('Wait for it...'):
        time.sleep(5)
    st.success('Boxplot shown!')
    feat_to_show = st.multiselect('Which feature do you want to show?', dist_columns)
    if feat_to_show:
        select_data = all_data[feat_to_show]
        select_data = pd.concat([select_data, all_data['TARGET']], axis=1)
        data_select_melt = select_data.melt(id_vars=['TARGET'])

        get_boxplot_comparison(data_select_melt, client, feat_to_show)

        hist_show = st.checkbox('Show distribution features')
        if hist_show:
            get_hist_comparison(feat_to_show, all_data, client)

    else:
        get_boxplot_comparison(data_melt_all, client, all_features)

        hist_show = st.checkbox('Show distribution features')
        if hist_show:
            get_hist_comparison(dist_columns, all_data, client)



