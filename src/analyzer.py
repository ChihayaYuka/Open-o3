import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results(result_dir):
   """
   Analyzes the JSON result files generated by the o3 reasoning framework.

   Args:
       result_dir: The directory containing the JSON result files.
   """

   results = []
   for filename in os.listdir(result_dir):
       if filename.endswith(".json") and filename.startswith("reasoner_result_"):
           filepath = os.path.join(result_dir, filename)
           try:
               with open(filepath, "r", encoding="utf-8") as f:
                   data = json.load(f)
                   results.append(data)
           except Exception as e:
               st.error(f"Error reading file {filename}: {e}")

   if not results:
       st.warning("No result files found in the specified directory.")
       return

   # Convert results to a Pandas DataFrame for easier analysis
   df = pd.DataFrame(results)

   # --- Basic Statistics ---
   st.header("Basic Statistics")

   # Token Usage
   st.subheader("Token Usage")
   st.write(df["token_usage"].describe())
   fig_token, ax_token = plt.subplots()
   sns.histplot(df["token_usage"], ax=ax_token, kde=True)
   ax_token.set_title("Token Usage Distribution")
   st.pyplot(fig_token)

   # Generation Time
   st.subheader("Generation Time (seconds)")
   # Handle missing generation_time_sec values gracefully
   generation_times = df["generation_time_sec"].dropna()  # Remove NaN values
   if not generation_times.empty:  # Check if there are any valid times
       st.write(generation_times.describe())
       fig_time, ax_time = plt.subplots()
       sns.histplot(generation_times, ax=ax_time, kde=True)
       ax_time.set_title("Generation Time Distribution")
       st.pyplot(fig_time)
   else:
       st.warning("No generation time data available (generation_time_sec is missing).")


   # --- Analysis of Math Metrics ---
   st.header("Analysis of Mathematical Metrics")

   # Relative Entropy
   st.subheader("Relative Entropy")
   # Extract relative entropy safely
   relative_entropies = df["math_metrics"].apply(lambda x: x.get("relative_entropy"))
   relative_entropies = relative_entropies.dropna()  # Remove NaN values
   if not relative_entropies.empty:
       st.write(relative_entropies.describe())
       fig_entropy, ax_entropy = plt.subplots()
       sns.histplot(relative_entropies, ax=ax_entropy, kde=True)
       ax_entropy.set_title("Relative Entropy Distribution")
       st.pyplot(fig_entropy)
   else:
       st.warning("No relative entropy data available (relative_entropy is missing).")

   # Complexity Value
   st.subheader("Complexity Value")
   # Extract complexity value safely
   complexity_values = df["math_metrics"].apply(lambda x: x.get("complexity_value"))
   complexity_values = complexity_values.dropna() # Remove NaN values
   if not complexity_values.empty:
       st.write(complexity_values.describe())
       fig_complexity, ax_complexity = plt.subplots()
       sns.histplot(complexity_values, ax=ax_complexity, kde=True)
       ax_complexity.set_title("Complexity Value Distribution")
       st.pyplot(fig_complexity)
   else:
       st.warning("No complexity value data available (complexity_value is missing).")

   # --- Text Analysis (Queries and Answers) ---
   st.header("Text Analysis")

   # Query Length
   st.subheader("Query Length")
   df["query_length"] = df["query"].apply(len)
   st.write(df["query_length"].describe())
   fig_query_length, ax_query_length = plt.subplots()
   sns.histplot(df["query_length"], ax=ax_query_length, kde=True)
   ax_query_length.set_title("Query Length Distribution")
   st.pyplot(fig_query_length)

   # Answer Length
   st.subheader("Answer Length")
   df["answer_length"] = df["best_answer"].apply(len)
   st.write(df["answer_length"].describe())
   fig_answer_length, ax_answer_length = plt.subplots()
   sns.histplot(df["answer_length"], ax=ax_answer_length, kde=True)
   ax_answer_length.set_title("Answer Length Distribution")
   st.pyplot(fig_answer_length)

   # --- Correlations ---
   st.header("Correlations")
   # Prepare the correlation matrix, handling missing data
   correlation_data = df[["token_usage", "generation_time_sec", "query_length", "answer_length"]].copy()
   correlation_data = correlation_data.dropna()  # Remove rows with any NaN values

   if not correlation_data.empty:
       corr = correlation_data.corr()
       fig_corr, ax_corr = plt.subplots()
       sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax_corr)
       ax_corr.set_title("Correlation Matrix")
       st.pyplot(fig_corr)
   else:
       st.warning("Not enough data to calculate correlations (missing values).")

   # --- Data Table ---
   st.header("Raw Data")
   st.dataframe(df)

# --- Streamlit UI ---
st.title("o3 Reasoning Framework Result Analyzer")

result_dir = st.text_input("Enter the directory containing the result files:", "./results")

if st.button("Analyze"):
   analyze_results(result_dir)
