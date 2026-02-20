import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # for better looking plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

#--- 1. Data Generation and Initial Setup ---
# Hum thoda zyada realistic dummy data banayenge
np.random.seed(42) # For reproducibility

study_hours = np.random.uniform(2, 12, 100) # 2 to 12 hours
previous_scores = np.random.uniform(30, 95, 100) # 30 to 95 marks
# Final marks will be a linear combination of study hours and previous scores, plus some noise
final_marks = 30 + (study_hours * 3.5) + (previous_scores * 0.4) + np.random.normal(0, 5, 100)
final_marks = np.clip(final_marks, 0, 100) # Ensure marks are between 0 and 100

data = pd.DataFrame({
    'Study_Hours': study_hours,
    'Prev_Scores': previous_scores,
    'Final_Marks': final_marks
})
 #--- 2. Model Training ---
X = data[['Study_Hours', 'Prev_Scores']]
y = data['Final_Marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation (for display purposes)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# --- Streamlit App ---
st.set_page_config(page_title="🎓 AI Study Helper", layout="wide")

st.title("AI Study Helper: Predict Your Exam Success")
st.markdown("""
Welcome to your personalized AI Study Helper! This tool uses a simple Machine Learning model
to predict your potential final exam marks based on your study habits and previous academic performance.
""")

st.sidebar.header("Model Performance")
st.sidebar.write(f"**Mean Absolute Error (MAE):** {mae:.2f} points")
st.sidebar.write(f"**R-squared (R²):** {r2:.2f}")
st.sidebar.info("MAE indicates the average absolute difference between predicted and actual marks. R² represents how well the model fits the data (1.0 is a perfect fit).")

st.header("Enter Your Details for Prediction")

col1, col2 = st.columns(2)

with col1:
    study_input = st.slider("How many hours do you plan to study?", min_value=1.0, max_value=20.0, value=8.0, step=0.5)
with col2:
    prev_score_input = st.slider("What was your average score in previous tests/quizzes?", min_value=0.0, max_value=100.0, value=75.0, step=1.0)

if st.button("Predict My Final Marks"):
    input_data = pd.DataFrame([[study_input, prev_score_input]], columns=['Study_Hours', 'Prev_Scores'])
    predicted_mark = model.predict(input_data)[0]

    st.subheader(f"Predicted Final Exam Marks: {predicted_mark:.2f}%")
    st.success("Remember, this is a prediction based on historical data. Your effort and understanding are key!")

    st.balloons() # Fun animation

st.markdown("---")

st.header("Data Insights & Visualizations")
st.markdown("Let's visualize the relationship between study hours, previous scores, and final marks based on the data used to train the model.")

# Scatter plot: Study Hours vs Final Marks
st.subheader("Study Hours vs. Final Marks")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Study_Hours', y='Final_Marks', data=data, hue='Prev_Scores', size='Prev_Scores', sizes=(20, 400), palette='viridis', ax=ax1)
ax1.set_title("Impact of Study Hours on Final Marks (Color by Previous Scores)")
ax1.set_xlabel("Study Hours")
ax1.set_ylabel("Final Marks (%)")
ax1.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig1)
st.markdown("This graph shows that generally, more study hours lead to higher marks. The color gradient indicates previous scores – typically, students with higher previous scores also tend to achieve higher final marks.")

# Scatter plot: Previous Scores vs Final Marks
st.subheader("Previous Scores vs. Final Marks")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Prev_Scores', y='Final_Marks', data=data, hue='Study_Hours', size='Study_Hours', sizes=(20, 400), palette='magma', ax=ax2)
ax2.set_title("Impact of Previous Scores on Final Marks (Color by Study Hours)")
ax2.set_xlabel("Previous Scores (%)")
ax2.set_ylabel("Final Marks (%)")
ax2.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig2)
st.markdown("Here, we can see a strong positive correlation: better previous scores are a good indicator of higher final marks. The study hours are also influencing this relationship.")

# Distribution of Final Marks
st.subheader("Distribution of Final Marks")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.histplot(data['Final_Marks'], kde=True, bins=15, color='skyblue', ax=ax3)
ax3.set_title("Distribution of Final Marks in the Dataset")
ax3.set_xlabel("Final Marks (%)")
ax3.set_ylabel("Number of Students")
ax3.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig3)
st.markdown("This histogram shows how the final marks are distributed across the dataset, giving us an idea of the typical performance range.")

st.markdown("---")
st.markdown("### How to Use This App Locally:")
st.code("""
1. Save the code above as a Python file (e.g., `study_helper_app.py`).
2. Open your terminal or command prompt.
3. Navigate to the directory where you saved the file.
4. Run the command: `streamlit run study_helper_app.py`
5. Your browser will automatically open the app!
""")

st.info("**Pro Tip:** This model is based on simplified data. In a real-world scenario, you'd collect data from many students over time, including factors like sleep, attendance, course difficulty, and more, to build a much more robust predictive model.") 