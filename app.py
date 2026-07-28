import joblib
import pandas as pd
import numpy as np

import streamlit as st
from PIL.ImageOps import expand
from streamlit_option_menu import option_menu

import plotly.express as px
import plotly.graph_objects as go

df=pd.read_csv("hr_emp2.csv")

#model load and setup
data=joblib.load("model.pk1")
model=data["model"]
scaler = data["scaler"]
feature_columns=data["feature_columns"]
threshold=data["default_threshold"]

input_data = pd.DataFrame(
    [[0] * len(feature_columns)],
    columns=feature_columns
)

#print(input_data)


#dashboard

with st.sidebar:
    option=option_menu("Main Menu",["Employee Attrition Dashboard","About Project"])

if option=="About Project":
    option2=option_menu("About",["Dataset","Project"],orientation="horizontal")

    if option2=="Dataset":
        ogdf=pd.read_csv("og_dataset.csv")
        st.title("Dataset")
        st.subheader("IBM HR Analytics Employee Attrition & Performance(Kaggle)")
        st.write(ogdf)
    elif option2 == "Project":
        st.title("Project")
        st.markdown("""
        # AI-Powered Employee Attrition Prediction & HR Decision Support System

        The **AI-Powered Employee Attrition Prediction & HR Decision Support System** is a machine learning application that helps Human Resource (HR) professionals identify employees who are at risk of leaving an organization. By analyzing employee demographics, workplace attributes, compensation, performance, and job-related factors, the system predicts the probability of employee attrition and provides intelligent decision support to assist HR teams in improving employee retention.

        Unlike traditional prediction systems, this application not only estimates an employee's likelihood of leaving but also explains **why** the prediction was made through feature contribution analysis and generates actionable HR recommendations based on both the predicted risk and the most influential contributing factors.

        ---

        ## 🎯 Project Objectives

        - Predict whether an employee is likely to stay or leave the organization.
        - Estimate the employee's probability of attrition.
        - Allow organizations to customize prediction sensitivity using an adjustable decision threshold.
        - Identify the key factors influencing each prediction.
        - Generate personalized HR recommendations for employee retention.
        - Support proactive, data-driven workforce management.

        ---

        ## 🚀 Key Features

        - 🤖 AI-powered employee attrition prediction
        - 📊 Interactive Streamlit dashboard
        - 🎯 Adjustable prediction threshold
        - 📈 Employee leave probability gauge
        - 📋 Employee profile input system
        - 📊 Feature contribution analysis
        - 📉 Top contributing factors visualization
        - 💡 Personalized HR recommendations
        - 🏢 Organization-specific decision support
        - ⚡ Real-time predictions using a trained ML model

        ---

        ## 🧠 Machine Learning Workflow

        1. Data Collection
        2. Data Preprocessing
        3. Feature Engineering
        4. Model Training
        5. Model Evaluation
        6. Model Deployment
        7. Real-Time Employee Prediction
        8. HR Decision Support Generation

        ---

        ## 📊 Employee Input Features

        The model evaluates multiple employee attributes, including:

        - Age
        - Gender
        - Department
        - Job Role
        - Business Travel
        - Education
        - Education Field
        - Marital Status
        - Monthly Income
        - Daily Rate
        - Hourly Rate
        - Monthly Rate
        - Job Level
        - Job Involvement
        - Job Satisfaction
        - Environment Satisfaction
        - Relationship Satisfaction
        - Work-Life Balance
        - Performance Rating
        - Stock Option Level
        - Percent Salary Hike
        - Distance From Home
        - Overtime
        - Training Times Last Year
        - Total Working Years
        - Years at Company
        - Years in Current Role
        - Years Since Last Promotion
        - Years With Current Manager
        - Number of Companies Worked

        ---

        ## 📈 Prediction & Decision Support

        The system provides:

        - ✅ Employee Stay / Leave Prediction
        - 📊 Leave Probability Score
        - 🎯 Organization-specific Prediction based on Adjustable Threshold
        - 📈 Risk Category Assessment
        - 📋 HR Retention Recommendations
        - 📊 Top Feature Contribution Analysis
        - 📉 Interactive Contribution Bar Chart
        - 💡 Personalized Recommendations based on Key Contributing Factors

        ---

        ## 💡 Intelligent HR Decision Support

        The dashboard assists HR professionals by:

        - Detecting employees at high risk of attrition.
        - Explaining the primary reasons behind each prediction.
        - Highlighting the most influential employee attributes.
        - Providing personalized retention strategies.
        - Supporting proactive HR intervention.
        - Helping optimize employee engagement and retention planning.

        ---

        ## 🛠️ Technologies Used

        - 🐍 Python
        - 🎈 Streamlit
        - 🤖 Scikit-learn
        - 🐼 Pandas
        - 🔢 NumPy
        - 💾 Joblib
        - 📈 Plotly
        - 📊 Matplotlib

        ---

        ## 👨‍💻 Developed By

        **Shriyans Mohanty**

        ---

        ## 📌 Conclusion

        This project demonstrates the practical application of **Artificial Intelligence**, **Machine Learning**, and **Explainable AI (XAI)** in Human Resource Management. By combining predictive analytics, feature contribution analysis, threshold-based decision making, and personalized HR recommendations, the system provides organizations with an intelligent decision support platform that enables proactive employee retention, reduces attrition-related costs, and supports informed HR decision-making.
        """)



elif option=="Employee Attrition Dashboard":
    st.sidebar.divider()

    st.sidebar.title("Decision Threshold")
    threshold = st.sidebar.slider(
        "Leave Prediction Threshold",
        min_value=0.05,
        max_value=0.90,
        value=float(threshold),
        step=0.01,
    )
    if threshold<=0.20:
        st.sidebar.markdown("🟢Aggressive Retention")
    elif threshold <= 0.35:
        st.sidebar.markdown("🟡Balanced")
    elif threshold <= 0.60:
        st.sidebar.markdown("🟠Conservative")
    elif threshold <= 0.90:
        st.sidebar.markdown("🔴High Confidence")

    st.sidebar.markdown("""
    **Threshold Strategy Guide**

    🟢 **0.05 – 0.20 | Aggressive Retention**
    - Flags more employees as high risk.
    - Higher recall (fewer leavers missed).
    - More HR interventions and higher retention costs.

    🟡 **0.21 – 0.35 | Balanced (Recommended)**
    - Good balance between recall and precision.
    - Suitable for most organizations.

    🟠 **0.36 – 0.60 | Conservative**
    - Flags fewer employees.
    - Lower intervention costs.
    - Some genuine leavers may be missed.

    🔴 **0.61 – 0.90 | High Confidence Only**
    - Only very high-risk employees are flagged.
    - Lowest intervention cost.
    - Highest chance of missing future leavers.
    """)
    st.sidebar.divider()


    st.sidebar.title("Employee Data Input")
    name = st.sidebar.text_input("Name of Employee",value="Default")
    st.sidebar.subheader("Numeric Data")
    age=st.sidebar.slider("Age",18,70,int(df["Age"].mean()))
    input_data["Age"]=age

    col1,col2= st.sidebar.columns(2)

    with col1:
        dailyRate=st.slider("Daily Rate",0,2000,int(df["DailyRate"].mean()))
        input_data["DailyRate"] = dailyRate

        dist_home=st.slider("Distance From Home(KM)",0,70,int(df["DistanceFromHome"].mean()))
        input_data["DistanceFromHome"] = dist_home

        edu=st.slider("Education Level(0-5)",0,5,int(df["Education"].mean()))
        input_data["Education"] = edu

        envs=st.slider("Environment Satisfaction Level(0-5)",0,5,int(df["EnvironmentSatisfaction"].mean()))
        input_data["EnvironmentSatisfaction"]=envs

        hourRate = st.slider("Hourly Rate", 0, 200, int(df["HourlyRate"].mean()))
        input_data["HourlyRate"] = hourRate

        jobinvolv = st.slider("Job Involvement Level(0-5)", 0, 5, int(df["JobInvolvement"].mean()))
        input_data["JobInvolvement"] = jobinvolv

        joblvl = st.slider("Job Level(0-5)", 0, 5, int(df["JobLevel"].mean()))
        input_data["JobLevel"] = joblvl

        jobsat = st.slider("Job Satisfaction Level (1-4)", 1, 4, int(df["JobSatisfaction"].mean()))
        input_data["JobSatisfaction"] = jobsat

        monthlyincome = st.slider("Monthly Income", 1000, 25000, int(df["MonthlyIncome"].mean()))
        input_data["MonthlyIncome"] = monthlyincome

        monthlyrate = st.slider("Monthly Rate", 2000, 30000, int(df["MonthlyRate"].mean()))
        input_data["MonthlyRate"] = monthlyrate

        numcompanies = st.slider("Number of Companies Worked", 0, 10, int(df["NumCompaniesWorked"].mean()))
        input_data["NumCompaniesWorked"] = numcompanies

    with col2:
        percentsalary = st.slider("Percent Salary Hike", 10, 30, int(df["PercentSalaryHike"].mean()))
        input_data["PercentSalaryHike"] = percentsalary

        performancerating = st.slider("Performance Rating", 1, 4, int(df["PerformanceRating"].mean()))
        input_data["PerformanceRating"] = performancerating

        relationshipsat = st.slider("Relationship Satisfaction Level (1-4)", 1, 4, int(df["RelationshipSatisfaction"].mean()))
        input_data["RelationshipSatisfaction"] = relationshipsat

        stockoption = st.slider("Stock Option Level", 0, 3, int(df["StockOptionLevel"].mean()))
        input_data["StockOptionLevel"] = stockoption

        totalworkingyears = st.slider("Total Working Years", 0, 40, int(df["TotalWorkingYears"].mean()))
        input_data["TotalWorkingYears"] = totalworkingyears

        trainingtimes = st.slider("Training Times Last Year", 0, 10, int(df["TrainingTimesLastYear"].mean()))
        input_data["TrainingTimesLastYear"] = trainingtimes

        worklifebalance = st.slider("Work-Life Balance (1-4)", 1, 4, int(df["WorkLifeBalance"].mean()))
        input_data["WorkLifeBalance"] = worklifebalance

        yearsatcompany = st.slider("Years at Company", 0, 40, int(df["YearsAtCompany"].mean()))
        input_data["YearsAtCompany"] = yearsatcompany

        yearsincurrentrole = st.slider("Years in Current Role", 0, 20, int(df["YearsInCurrentRole"].mean()))
        input_data["YearsInCurrentRole"] = yearsincurrentrole

        yearssincelastpromotion = st.slider("Years Since Last Promotion", 0, 15, int(df["YearsSinceLastPromotion"].mean()))
        input_data["YearsSinceLastPromotion"] = yearssincelastpromotion

        yearswithcurrmanager = st.slider("Years With Current Manager", 0, 20, int(df["YearsWithCurrManager"].mean()))
        input_data["YearsWithCurrManager"] = yearswithcurrmanager


        st.sidebar.subheader("Categorical Data")
        col3,col4=st.sidebar.columns(2)

        with col3:
            gender=st.selectbox("Gender",[0,1],format_func=lambda x: "Male" if x else "Female")
            input_data["Gender"]=gender

            b_travel=st.selectbox("Business Travel Frequency",["Non-Travel","Travel Rarely","Travel Frequently"])
            if b_travel=="Travel Rarely":
                input_data["BusinessTravel_Travel_Rarely"]=1
            elif b_travel=="Travel Frequently":
                input_data["BusinessTravel_Travel_Frequently"]=1

            dept=st.selectbox("Department",["Research & Development","Sales","Human Resources"])
            if dept=="Research & Development":
                input_data["Department_Research & Development"] = 1
            elif dept=="Sales":
                input_data["Department_Sales"] = 1

            edu_f = st.selectbox("Education Field", ["Life Sciences", "Medical", "Technical Degree","Human Resources","Marketing","Other"])
            if edu_f=="Life Sciences":
                input_data["EducationField_Life Sciences"]=1
            elif edu_f=="Medical":
                input_data["EducationField_Medical"] = 1
            elif edu_f=="Technical Degree":
                input_data["EducationField_Technical Degree"] = 1
            elif edu_f=="Marketing":
                input_data["EducationField_Marketing"] = 1
            elif edu_f=="Other":
                input_data["EducationField_Other"] = 1


        with col4:
            job_role = st.selectbox(
                "Job Role",
                [
                    "Healthcare Representative",
                    "Human Resources",
                    "Laboratory Technician",
                    "Manager",
                    "Manufacturing Director",
                    "Research Director",
                    "Research Scientist",
                    "Sales Executive",
                    "Sales Representative"
                ]
            )

            if job_role == "Human Resources":
                input_data["JobRole_Human Resources"] = 1
            elif job_role == "Laboratory Technician":
                input_data["JobRole_Laboratory Technician"] = 1
            elif job_role == "Manager":
                input_data["JobRole_Manager"] = 1
            elif job_role == "Manufacturing Director":
                input_data["JobRole_Manufacturing Director"] = 1
            elif job_role == "Research Director":
                input_data["JobRole_Research Director"] = 1
            elif job_role == "Research Scientist":
                input_data["JobRole_Research Scientist"] = 1
            elif job_role == "Sales Executive":
                input_data["JobRole_Sales Executive"] = 1
            elif job_role == "Sales Representative":
                input_data["JobRole_Sales Representative"] = 1

            marital = st.selectbox(
                "Marital Status",
                ["Divorced", "Married", "Single"]
            )

            if marital == "Married":
                input_data["MaritalStatus_Married"] = 1
            elif marital == "Single":
                input_data["MaritalStatus_Single"] = 1

            overtime = st.selectbox(
                "OverTime",
                [0, 1],format_func=lambda x: "Yes" if x else "No")

            input_data["OverTime"] = overtime


    input_data_scaled=scaler.transform(input_data)
    leave_prob=model.predict_proba(input_data_scaled)[0][1]
    #st.write(leave_prob)


    st.set_page_config(page_title="HR EMPLOYEE ATTRITION PREDICTION SYSTEM",layout="wide")
    st.title("HR EMPLOYEE ATTRITION PREDICTION SYSTEM")
    st.subheader("Machine Learning Based Employee Attrition Prediction & HR Decision Support System👤")
    st.markdown("#### ML Training Based on the IBM HR Analytics Employee Attrition & Performance(Kaggle)")
    st.markdown("###### Developed by: Shriyans Mohanty")

    st.divider()

    st.header(f"Employee Name: {name}")

    st.divider()

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=leave_prob * 100,
        number={"suffix": "%"},

        # title={"text": "Predicted Risk Score"},

        gauge={
            "axis": {"range": [0, 100]},

            "bar": {"color": "#0B3C5D"},  # Dark Blue needle/bar

            "steps": [
                {"range": [0, 33], "color": "#2E8B57"},  # Sea Green
                {"range": [33, 66], "color": "#D4A017"},  # Golden
                {"range": [66, 100], "color": "#C0392B"}  # Dark Red
            ]
        }

    ))

    st.subheader("Employee Leave Probability")
    st.plotly_chart(fig, use_container_width=True)
    leave_val=leave_prob * 100
    if leave_val < 33:
        chance_level = "🟢 LOW CHANCES"
        color = "#2E8B57"
    elif leave_val < 66:
        chance_level = "🟡 MEDIUM CHANCES"
        color = "#D4A017"
    else:
        chance_level = "🔴 HIGH CHANCES"
        color = "#C0392B"

    st.markdown(
        f"""
               <div style='text-align:center; margin-top:-50px;'>
                   <span style='font-size:20px; font-weight:700; color:{color};'>
                       {chance_level}
                   </span>
               </div>
               <br>
               """,
        unsafe_allow_html=True
    )

    st.subheader("Company based prediction (According To Set Threshold)")

    diff=leave_prob-threshold
    #st.write(diff)

    if diff >= 0.30:
        risk = "🔴 Critical Risk"
        st.error("#### General Recommendations")
        st.markdown("""
    - 🚨 Immediate HR intervention is strongly recommended.
    - 👥 Schedule a one-on-one discussion with the employee.
    - 💰 Review salary, incentives, and benefits.
    - 📈 Evaluate promotion or career growth opportunities.
    - ⚖️ Reduce excessive workload or overtime where applicable.
    - 🤝 Assign a mentor or senior manager for regular follow-ups.
    - 📅 Monitor employee engagement on a weekly basis.
    """)

    elif diff >= 0.15:
        risk = "🔴 High Risk"
        st.error("#### General Recommendations")
        st.markdown("""
    - 👥 Arrange an HR discussion within the next few days.
    - 😊 Assess employee satisfaction and workplace concerns.
    - ⚖️ Review work-life balance and overtime.
    - 📚 Explore training and career development opportunities.
    - 🤝 Encourage regular manager check-ins.
    - 📅 Monitor employee progress monthly.
    """)

    elif diff >= 0.05:
        risk = "🟡 Moderate Risk"
        st.warning("#### General Recommendations")
        st.markdown("""
    - 📊 Conduct an employee satisfaction survey.
    - 📝 Review recent performance feedback.
    - 📚 Encourage participation in learning and development programs.
    - 👥 Increase employee engagement initiatives.
    - 📅 Continue monitoring during upcoming review cycles.
    """)

    elif diff >= -0.05:
        risk = "🟠 Borderline Risk"
        st.warning("#### General Recommendations")
        st.markdown("""
    - ⚠️ Employee is close to the decision threshold.
    - 👥 Schedule periodic manager check-ins.
    - 📊 Monitor job satisfaction and engagement.
    - 📅 Reassess employee status during the next performance review.
    - 🔍 No immediate intervention is required, but close observation is recommended.
    """)

    elif diff >= -0.20:
        risk = "🟢 Low Risk"
        st.success("#### General Recommendations")
        st.markdown("""
    - ✅ Continue regular employee engagement.
    - 🏆 Recognize good performance and achievements.
    - 📚 Offer optional training and career development opportunities.
    - 📅 Monitor through standard HR review cycles.
    """)

    else:
        risk = "🟢 Very Low Risk"
        st.success("#### General Recommendations")
        st.markdown("""
    - ✅ No immediate HR intervention is required.
    - 🌱 Maintain the current work environment and policies.
    - 🏆 Continue employee recognition and growth opportunities.
    - 📅 Review during routine performance evaluations only.
    """)

    st.subheader(risk)

    st.divider()
    st.markdown("#### Highest Contributing Factors")
    contri = (model.coef_.flatten() * input_data_scaled[0])*100

    contri_df = pd.DataFrame({
        "Feature": feature_columns,
        "Contribution": contri
    })

    contri_df = contri_df.reindex(
        contri_df["Contribution"].abs().sort_values(ascending=False).index
    )
    contri_df=contri_df[contri_df["Contribution"]>0]
    contri_df= contri_df.head(5)

    fig1=px.bar(contri_df,y="Feature",x="Contribution",orientation="h")
    st.plotly_chart(fig1,use_container_width=True)
    #st.subheader("Highest Contributing Factors")
    #st.dataframe(contri_df["Feature"])

    recommendations = {
        "OverTime": "• Consider reducing overtime or redistributing workload to improve work-life balance.",

        "MonthlyIncome": "• Review compensation and salary competitiveness for this employee.",

        "JobSatisfaction": "• Conduct a job satisfaction discussion and identify workplace concerns.",

        "EnvironmentSatisfaction": "• Improve the employee's work environment and address workplace issues.",

        "WorkLifeBalance": "• Encourage flexible work arrangements and promote a healthier work-life balance.",

        "DistanceFromHome": "• Consider remote work, hybrid work, or relocation assistance if feasible.",

        "YearsSinceLastPromotion": "• Evaluate promotion eligibility and provide a clear career progression plan.",

        "YearsInCurrentRole": "• Discuss role enrichment, internal mobility, or new responsibilities.",

        "YearsWithCurrManager": "• Review manager-employee relationship and leadership effectiveness.",

        "RelationshipSatisfaction": "• Encourage team-building activities and improve workplace relationships.",

        "TrainingTimesLastYear": "• Provide additional training and professional development opportunities.",

        "BusinessTravel_Travel_Frequently": "• Reduce travel frequency where possible or provide additional travel support.",

        "BusinessTravel_Travel_Rarely": "• Review whether travel expectations align with employee preferences.",

        "StockOptionLevel": "• Consider enhancing long-term incentives such as stock options or retention bonuses.",

        "JobInvolvement": "• Increase employee engagement through meaningful projects and decision-making opportunities.",

        "PerformanceRating": "• Recognize achievements and provide constructive performance feedback.",

        "PercentSalaryHike": "• Review recent salary adjustments and ensure compensation remains competitive.",

        "NumCompaniesWorked": "• Discuss long-term career goals and emphasize organizational growth opportunities.",

        "TotalWorkingYears": "• Align responsibilities with the employee's experience and career aspirations.",

        "Age": "• Ensure career development opportunities are appropriate for the employee's career stage.",

        "Department_Sales": "• Monitor workload, sales targets, and employee burnout within the sales department.",

        "Department_Research & Development": "• Encourage innovation, recognition, and technical growth opportunities.",

        "Department_Human Resources": "• Review workload distribution and employee engagement initiatives.",

        "JobRole_Sales Executive": "• Monitor sales pressure, incentives, and customer workload.",

        "JobRole_Laboratory Technician": "• Improve workplace conditions and provide technical skill development.",

        "JobRole_Research Scientist": "• Support research opportunities and recognize innovation.",

        "MaritalStatus_Single": "• Promote social engagement and work-life balance initiatives."
    }

    st.divider()

    st.markdown("#### Contribution Based Recommendations")

    for feature in contri_df.head(4)["Feature"]:
        if feature in recommendations:
            st.success(f"**{feature}**")
            st.write(recommendations[feature])