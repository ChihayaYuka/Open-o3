import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from o3_reasoner import o3  # Import the o3 class

# Initialize o3 reasoner with your desired settings
system_prompt = "您是由流明智能开发的大型推理模型 Open-o3."
reasoner = o3(system_prompt=system_prompt, enable_tda=False, save_results=False)

# Streamlit UI
st.title("o3 Reasoner LangChain App")

# User input
query = st.text_input("Enter your query:", "证明不存在最大的素数")

# Button to trigger reasoning
if st.button("Reason"):
   with st.spinner("Reasoning..."):
       # Call the o3 reasoner's predict method
       try:
           output = reasoner.predict(query, batch_size=8, plot_hist=False)

           # Display the results
           st.subheader("Answer:")
           st.write(output["answer"])

           st.subheader("Details:")
           st.write(f"Token Usage: {output['token_usage']}")
           st.write(f"Generation Time: {output['run_stats'].get('generation_time_sec', 'N/A')} seconds")
           st.write("Mathematical Metrics:")
           st.write(output["math_metrics"])

           # Display a detailed report (optional)
           st.subheader("Detailed Report:")
           reasoner.detailed_report()

       except Exception as e:
           st.error(f"An error occurred: {e}")
