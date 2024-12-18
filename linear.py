import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Title of the app
st.title("📈 Linear Regression App")

# Instructions
st.write("""
### Upload your CSV file and select features for Linear Regression.
- Choose the **Feature** (X) and **Label** (y) dynamically.
- Train a Linear Regression model and visualize the results.
""")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data into a DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset:")
    st.write(df.head())

    # Allow user to select Feature (X) and Target (y)
    st.write("### Select Feature and Target:")
    feature = st.selectbox("Select the Feature (X):", df.columns, key="feature")
    target = st.selectbox("Select the Target (y):", df.columns, key="target")

    if feature and target:
        # Define X (feature) and y (target)
        X = df[[feature]]
        y = df[target]

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Model Evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        coef = model.coef_[0]
        intercept = model.intercept_

        # Display Results
        st.write("### Model Evaluation:")
        st.write(f"**Coefficient (Slope):** {coef}")
        st.write(f"**Intercept:** {intercept}")
        st.write(f"**Mean Squared Error (MSE):** {mse}")
        st.write(f"**R-squared (R²):** {r2}")

        # Visualization
        st.write("### Visualization of Regression Line:")
        fig, ax = plt.subplots()
        ax.scatter(X_test, y_test, color='blue', label='Actual Data')
        ax.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.set_title("Linear Regression: Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to proceed.")
