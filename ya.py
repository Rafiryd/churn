import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    st.title("Customer Churn Prediction App")

    # File uploader for selecting the data file
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Read the uploaded data file
        data = pd.read_csv(uploaded_file)

        # Drop unnecessary columns
        data.drop(["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
                   "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"],
                  axis=1, inplace=True)

        # Convert categorical variables to numerical using label encoding
        categorical_cols = ["Attrition_Flag", "Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]
        le = LabelEncoder()
        for col in categorical_cols:
            data[col] = le.fit_transform(data[col])

        # Split the data into training and testing sets
        X = data.drop("Attrition_Flag", axis=1)
        y = data["Attrition_Flag"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply SMOTE
        smote = SMOTE()
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Apply PCA
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_smote)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=0.95)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        # Apply Clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(X_train_pca)

        # Build Churn Prediction Model
        lr = LogisticRegression()
        lr.fit(X_train_pca, y_train_smote)

        # Make Predictions
        y_pred = lr.predict(X_test_pca)

        # Evaluate the model's performance
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Display the results
        st.header("Churn Prediction Model Performance:")
        st.write("Accuracy:", accuracy)
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("F1-Score:", f1)

        # Show the processed data
        st.subheader("Processed Data:")
        st.write(data)

        # Show the clusters
        st.subheader("Clusters:")
        st.write(kmeans.labels_)

        # Show sample predictions
        st.subheader("Sample Predictions:")

        for i in range(3):
            sample_data = X_test.sample(1)
            sample_data_pca = pca.transform(scaler.transform(sample_data))
            sample_prediction = lr.predict(sample_data_pca)[0]

            st.write("Sample", i+1)
            st.write("Input Data:")
            st.write(sample_data)
            st.write("Prediction:", sample_prediction)
            st.write("---")

        # Plotting the correlation matrix
        st.subheader("Correlation Matrix:")
        corr_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        st.pyplot()

if __name__ == "__main__":
    main()
