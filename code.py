#pip install pandas seaborn matplotlib gradio ollama
#Run the python program python code.py from command prompt
#This is with out 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gradio as gr
import ollama

# Load Titanic Dataset
url = r"titanic_ dataset_final.csv" #Add your path here where 
df = pd.read_csv(url)
df.head()

# Display dataset info
print(df.describe())

# Missing Values Check
print("\nMissing Values:\n", df.isnull().sum())

# Survival Rate Visualization
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

import ollama

def generate_insights(df_summary):
    prompt = f"Analyze the dataset summary and provide insights:\n\n{df_summary}"
    response = ollama.chat(model="gemma3:270m", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# Generate AI Insights
summary = df.describe().to_string()
insights = generate_insights(summary)
print("\n🔹 AI-Generated Insights:\n", insights)



def eda_analysis(file):
    df = pd.read_csv(file.name)
    summary = df.describe().to_string()
    insights = generate_insights(summary)
    return insights

# Create Web Interface
demo = gr.Interface(fn=eda_analysis, inputs="file", outputs="text", title="AI-Powered EDA with Mistral")

# Launch App
demo.launch(share=True)  # Use share=True for Google Colab