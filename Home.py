    
import  streamlit_antd_components as sac
import streamlit as st
import pandas as pd
import numpy as np
from my_module import *
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from config.config import Config
import plotly.express as px
import requests
import json



# Set page


st.set_page_config(page_title='Lokasi Insight', page_icon=f"./data/Image_ic.ico", layout='wide')

# Hide button at the top right of the page
hide_button()


with st.sidebar:
# Menu in Sidebar
    item = sac.menu([
        sac.MenuItem('Overview', icon='house-fill'),
        sac.MenuItem('Profile', icon="bi bi-person-lines-fill"),
        sac.MenuItem('Analytics', icon="bi bi-clipboard-data-fill"),
        sac.MenuItem('Predictions', icon="bi bi-search")],
          format_func='title', open_all=False, open_index=0, size=14)

if item =='Overview':
 
    # Project Overview
    st.markdown("""
    ### **Project Overview: Ads Click Prediction with Machine Learning**

    Welcome to the **Ads Click Prediction Application**! In today’s digital age, understanding user behavior and predicting the likelihood of an advertisement being clicked is crucial for optimizing online advertising campaigns. This application uses advanced machine learning techniques to accurately predict the likelihood of an advertisement being clicked based on various user characteristics and behavior patterns.

    ### **What does this application do?**

    This application predicts whether an advertisement will be clicked by a user based on a variety of user-specific factors. By analyzing features such as:

    - **Time Spent on Site**: The duration a user spends on the website, which can indicate their level of engagement.
    - **Age**: The user's age, which may influence the type of content or ads they find appealing.
    - **Area Income**: The income level of the user's geographical region, helping to tailor ads based on socioeconomic factors.
    - **Daily Internet Usage**: The average number of hours a user spends online, which can help identify more frequent internet users who may be more likely to interact with ads.
    - **Gender**: Whether the user is male or female, which may influence the ads they are most likely to interact with.
    - **Device Information**: Data regarding the device used to access the site, which could affect how ads are presented.

    ### **How does the model work?**

    Using these features, we train a machine learning model to classify user interactions with displayed ads. The model learns from historical data, including past user clicks and interactions with ads, to identify patterns and predict whether a user will click on an ad.

    Our model applies **Logistic Regression**, a simple and interpretable classification algorithm that predicts the probability of an ad click. Logistic regression is well-suited for binary classification tasks, such as predicting whether a user will or will not click an ad, and provides probabilities that help quantify the likelihood of interaction.


    ### **Key Features of the Application:**

    - **Real-time predictions**: Input your user data, and get an immediate prediction on whether an ad is likely to be clicked or not.
    - **Visualization**: Interactive charts and graphs to help users understand the feature importance and prediction outcomes.
    - **Model Performance**: A detailed performance evaluation of the machine learning model with metrics like accuracy, precision, recall, and F1-score, ensuring that the predictions are trustworthy.
    - **Feature Importance**: Visual representation of the most important features influencing the model's predictions (e.g., age, daily internet usage, etc.), allowing users to understand the factors driving ad clicks.

    ### **What will you find in this application?**

    - **Data Exploration**: An exploration of the user behavior dataset, visualizations of feature distributions, and correlation analysis.
    - **Model Training**: Training and testing machine learning models to predict ad clicks with metrics for evaluation.
    - **Prediction Results**: Instant predictions based on user input, helping businesses make data-driven decisions in their ad strategies.
    - **Performance Metrics**: Key model performance metrics such as accuracy, precision, recall, and F1 score, ensuring transparency in model reliability.

    """)


if item =='Profile':
    

    st.markdown("""
    <div style="text-align: center;">
        <h4 style="font-weight:bold; margin-bottom: 0px; padding-bottom: 4px; padding-top: 0px;">Welly Oktariana, S.T</h3>
        <h5 style="font-weight:bold; margin-top: 0px; padding-bottom: 0px; padding-top: 0px;">Data Analyst | Data Science | Machine Learning</h4>
        <ppadding-bottom: 0px; padding-top: 0px;>South Tangerang, Banten | 0822-4767-8101 | <a href="mailto:wellyoktariana08@gmail.com">wellyoktariana08@gmail.com</a> | <a href="https://www.linkedin.com/in/wellyoktariana/" target="_blank">LinkedIn</a></p>
    </div>
    <div style ="font-weight:bold; text-align: left; margin-left:200px; margin-top:50px;">SUMMARY</div>
    <div style="font-weight: bold; text-align: left; margin-left: 200px; margin-right:200px; margin-top: 0px; border-top: 2px solid grey; width: 65%;"></div>
    <div style ="text-align: left; font-size:16px; margin-left:200px; margin-right:200px; margin-top:0px;">Data Analyst with 1+ year of experience in building machine learning models, performing spatial analysis, and leveraging data visualization techniques to deliver actionable insights that support strategic decision-making and drive business outcomes.</div>
    <div style ="font-weight:bold; text-align: left; margin-left:200px; margin-top:30px;">PROFESSIONAL EXPERIENCES</div>
    <div style="font-weight: bold; text-align: left; margin-left: 200px; margin-right:200px; margin-top: 0px; border-top: 2px solid grey; width: 65%;"></div>
    <div style="font-weight:bold; text-align: left; margin-left:200px; margin-top:5px;">
        <h6>PT. Bhumi Varta Technology</h6>
        <p><strong>Data Analyst</strong> <em>(May 2023 - Present)</em></p>
        <ul>
            <li>Conducted detailed analysis of datasets to identify key trends, patterns, and insights, driving data-driven decisions.</li>
            <li>Developed and implemented machine learning models (e.g., regression, classification) to solve business challenges and improve outcomes.</li>
            <li>Created interactive dashboards and visual reports to effectively communicate insights and performance metrics to stakeholders.</li>
        </ul>
        <h6>GIS Technician <em>(Feb - Apr 2023)</em></h5>
        <ul>
            <li>POI data preparation, concept and method initiation, and team training.</li>
            <li>Monitoring, discussion, and quality control of data validation results.</li>
        </ul>
        <h6>GIS Operator <em>(Oct 2021 - Jan 2023)</em></h6>
        <ul>
            <li>Collecting and cleaning POI data, validation of POI data, POI sampling accuracy, complete depth POI data, and digitization.</li>
        </ul>
        <h6>Badan Pendapatan Daerah DKI Jakarta</h6>
        <p><strong>Mapping Surveyor</strong> <em>(Oct 2020 - Sep 2021)</em></p>
        <ul>
            <li>Coordinated PBB mapping activities in Palmerah District, including verifying and processing PBB-P2 tax data, and providing progress reports to the Field and Regional Managers.</li>
        </ul>
    </div>""", unsafe_allow_html=True)
                
        
                


if item =='Analytics':
    st.markdown(f"""
                <div id="custom_legend" style="
                    position: relative;
                    z-index: 1000;
                    display: flex;
                    flex-direction: column;
                    align-items: flex-start;
                    text-align: left;
                    ">
                    <div style="display: flex; align-items: right; margin-bottom: 5px;">
                        <span style="font-size:35px; font-weight:bold; margin-top:0px; margin-bottom:20px;">Data Analytic and Model Performance </span>
                    </div>
                </div>""", unsafe_allow_html=True)
        
    tab = st.tabs(['Data Analytics','Model Performance'])
    with tab[0]:

        df = load_data()
        metrics, feature_importance = load_model_artifacts()
        st.markdown("""<p style="font-size:22px; font-weight:bold; margin-bottom:20px;">Sample Data</p>""", unsafe_allow_html=True)
        with st.container(border=True):

            # Define numerical columns
            st.dataframe(df.head(5))
        st.text('')
        st.text('')


        st.markdown("""<p style="font-size:22px; font-weight:bold; margin-bottom:20px;">Distiburion Analysis</p>""", unsafe_allow_html=True)
        with st.container(border=True):
            numerical_cols = ['Daily_Time_Spent_on_Site', 'Age', 'Area_Income',  'Daily_Internet_Usage']

            # Create subplots in 3 columns layout
            fig, axes = plt.subplots(1, 4, figsize=(18, 5))

            # Plot histograms for each numerical column
            for i, col in enumerate(numerical_cols[:4]):  # Only take the first 3 columns
                sns.histplot(df[col], kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')

            # Adjust layout to prevent overlapping
            plt.tight_layout()
            st.pyplot(fig)

                # Create subplots in 3 columns layout
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))

            col_grap =st.columns ([6,4])
            with col_grap[0]:
                col_cat = ['Male','Clicked_on_Ad']
                # Plot histograms for each numerical column
                for i, col in enumerate(col_cat[:2]):  # Only take the first 3 columns
                    df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'], ax=axes[i])
                    axes[i].set_title(f'Pie Chart of {col}')
            
                

                # Adjust layout to prevent overlapping
                plt.tight_layout()
                st.pyplot(fig)
            with col_grap[1]:
    
                # Hourly analysis based on 'Timestamp'
                fig =plt.figure(figsize=(12, 6))
                sns.histplot(df['Hour'][df['Clicked_on_Ad'] == 1], color='skyblue', kde=True, label='Clicked')
                sns.histplot(df['Hour'][df['Clicked_on_Ad'] == 0], color='salmon', kde=True, label='Not Clicked')
                plt.title('Distribution of Clicked on Ad by Hour')
                plt.legend()
                plt.show()
                st.pyplot(fig)


        st.text('')
        st.text('')

        st.markdown("""<p style="font-size:22px; font-weight:bold; margin-bottom:20px;">Correlation Analysis</p>""", unsafe_allow_html=True)
        with st.container(border=True):
            # Calculate correlation matrix
            cola = st.columns([5,5])
            with cola[0]:
                corr_matrix = df[Config.FEATURE_COLUMNS + [Config.TARGET_COLUMN]].corr()
                
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

                # Create the plot
                fig = plt.figure(figsize=(6, 4))

                # Plot the heatmap with a blue color scheme
                sns.heatmap(corr_matrix, annot=True, cmap='GnBu', fmt=".2f", mask=mask, cbar_kws={'label': 'Correlation Coefficient'})
                # Get the color bar and set the font size of the ticks
                colorbar = plt.gca().collections[0].colorbar
                colorbar.ax.tick_params(labelsize=8)  # Adjust the label font size here
                # Show the plot in Streamlit
                st.pyplot(fig)

            with cola[1]:
                col1, col2 = st.columns(2)
                with col1:
                    x_feature = st.selectbox("Select X-axis feature", Config.FEATURE_COLUMNS)
                with col2:
                    y_feature = st.selectbox(
                        "Select Y-axis feature", 
                        [Config.TARGET_COLUMN], 
                        index=1 if len([Config.TARGET_COLUMN]) > 1 else 0
                    )
                
                fig = px.scatter(
                    df,
                    x=x_feature,
                    y=y_feature,
                    title=f"{x_feature} vs {y_feature}",
                    trendline="ols"
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab[1]:
        st.markdown("""<p style="font-size:22px; font-weight:bold; margin-bottom:20px;">Model Performance</p>""", unsafe_allow_html=True)
        

        metrics = metrics["metrics"]
        # Convert the list of metrics to a dictionary
        metrics_dict = {}
        for metric in metrics:
            key, value = metric.split(": ")
            metrics_dict[key] = float(value)

        # Display metrics in Streamlit columns
        colmetrics = st.columns(4)
        colmetrics[0].metric("Accuracy", metrics_dict['Accuracy'])
        colmetrics[1].metric("Precision", metrics_dict['Precision'])
        colmetrics[2].metric("Recall", metrics_dict['Recall'])
        colmetrics[3].metric("F1 Score", metrics_dict['F1 Score'])
        st.text('')
        st.text('')

        
        col_fi = st.columns ([5,5])
        with col_fi[0]:
            feature_importance=feature_importance[0]
            # Create DataFrame from feature importance
            importance_df = pd.DataFrame({
                'Feature': list(feature_importance.keys()),
                'Importance': list(feature_importance.values())
            }).sort_values('Importance', ascending=True)

            # Horizontal bar chart
            fig = go.Figure(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker=dict(
                    color='rgb(26, 118, 255)',
                    line=dict(color='rgba(26, 118, 255, 1.0)', width=1)
                )
            ))

            fig.update_layout(
                title='Feature Importance',
                xaxis_title='Importance Score',
                yaxis_title='Features',
                template='plotly_white',
                height=400,
                margin=dict(l=0, r=0, t=30, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)
        with col_fi[1]:
        # Feature Importance Details
            importance_df_copy = importance_df.copy(deep=True)
            importance_df_copy['Importance'] = importance_df_copy['Importance'].abs()

            fig = px.pie(
                importance_df_copy,
                values='Importance',
                names='Feature',
                title='Feature Importance Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)


if item =='Predictions':

 

        # Prediction Form
        with st.form("prediction_form"):
            st.subheader("Enter Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                dailytime = st.slider(
                    "Daily Time Spent on Site",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    help=Config.FEATURE_DESCRIPTIONS['Daily_Time_Spent_on_Site']
                )
                
                age = st.slider(
                    "Age",
                    min_value=15,
                    max_value=80,
                    value=25,
                    help=Config.FEATURE_DESCRIPTIONS['Age']
                )
                
                income = st.slider(
                    "Area Income",
                    min_value=10000.0,
                    max_value=100000.0,
                    value=50000.0,
                    step=1000.0,
                    help=Config.FEATURE_DESCRIPTIONS['Area_Income']
                )
            
            with col2:
                internetusage = st.slider(
                    'Daily Internet Usage',
                    min_value=100.0,
                    max_value=300.0,
                    value=50.0,
                    step =5.0,
                    help=Config.FEATURE_DESCRIPTIONS['Daily_Internet_Usage']
                )
                
                male = st.slider(
                    "Male",
                    min_value=0,
                    max_value=1,
                    value=0,
                    help=Config.FEATURE_DESCRIPTIONS['Male']
                )
                
                hour= st.slider(
                    "Hour",
                    min_value=0,
                    max_value=24,
                    value=21,
                    help=Config.FEATURE_DESCRIPTIONS['Hour']
                )
            
            with col3:
                day = st.slider(
                    "Day of Week",
                    min_value=1,
                    max_value=7,
                    value=2,
                    help=Config.FEATURE_DESCRIPTIONS['DayOfWeek']
                )
                

            st.text('')
            st.text('')
            submitted = st.form_submit_button("Ads click Prediction", type='primary',use_container_width=True)

        if submitted:
            input_data = {
                "Daily_Time_Spent_on_Site":dailytime,
                "Age": age,
                "Area_Income": income,
                "Daily_Internet_Usage": internetusage,
                "Male": male,
                "Hour": hour,
                "DayOfWeek": day
            }

        # Update API endpoint URL untuk Docker
        # API_URL = "http://fastapi:8000"  # Gunakan nama service dari docker-compose
            
            # try:
            with st.spinner('Making prediction...'):
                try:
                    response = requests.post(
                        "http://localhost:8000/predict",
                        json=input_data
                    )
                    
                    
                    if response.status_code == 200:
                        prediction = response.json()["prediction"]
                        
                        if prediction == 0:
                            note ='(Ad will not be clicked)'
                        elif prediction == 1:
                           note ='(Ad will be clicked)'

                        st.success(f"#### Ads Click Prediction: {int(prediction)} {note}")
                    else:
                        st.error(f"Error making prediction: {response.text}")
                except:
                    import pickle
                    with open(Config.MODEL_PATH, 'rb') as model_file:
                        model = pickle.load(model_file)
                    
                    with open(Config.SCALER_PATH, 'rb') as model_file:
                        scaler = pickle.load(model_file)
                    
                    df_test = pd.DataFrame(input_data,index=[0])
                    col_transform = df_test.drop(columns='Male').columns.tolist()
                    df_test[col_transform] =scaler.transform(df_test[col_transform])  

                    prediction = model.predict(df_test)
                    if prediction == 0:
                        note ='(Ad will not be clicked)'
                    elif prediction == 1:
                           note ='(Ad will be clicked)'

                    st.success(f"#### Ads Click Prediction: {int(prediction)} {note}")




                        
                        


       