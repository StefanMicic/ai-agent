import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

input_folder = "./data/general_answering_data"  
output_folder = "./data/insight_direction_action_data"  
os.makedirs(output_folder, exist_ok=True)

model_name = "gpt-4o-mini"

def extract_data(file_content):
    system_message = (
        "You are a text analysis assistant specialized in extracting structured information. "
        "Your task is to analyze the provided text and identify and extract the following components: "
        "1. Insights: Key observations, metrics, trends, and issues described in the text. "
        "2. Direction: Recommended strategies or plans proposed to address the issues. "
        "3. Action: Specific actions or assignments provided to team members or departments. "
        "Output your findings under clearly labeled sections: 'Insights', 'Direction', and 'Action'."
    )
    prompt = (
        "Please analyze the following text and extract the following elements:\n"
        "- **Insights:** Identify any data points, observations, or trends mentioned.\n"
        "- **Direction:** Extract the recommendations or strategies that have been suggested.\n"
        "- **Action:** List any specific actions or assignments given.\n\n"
        "Input Text:\n"
        f"{file_content}\n\n"
        "Provide your answer in three separate sections labeled 'Insights', 'Direction', and 'Action', "
        "using bullet points for each extracted item."
    )

    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print("Error during OpenAI API call:", e)
        return None
    
def convert_csv_to_text(csv_path):
    """
    Reads a CSV file and converts it into a text summary.
    You can customize this function to summarize trends or include descriptive statistics.
    """
    try:
        df = pd.read_csv(csv_path)
        table_text = df.to_markdown(index=False)
        summary = f"CSV File Summary:\n- Total Records: {len(df)}\n- Columns: {', '.join(df.columns)}\n\n"
        return summary + table_text
    except Exception as e:
        print(f"Error processing CSV file {csv_path}: {e}")
        return None

def process_files(input_folder, output_folder):
    for index, filename in enumerate(os.listdir(input_folder)):
        filepath = os.path.join(input_folder, filename)
        file_content = ""
        
        if filename.lower().endswith(".csv"):
            print(f"Processing CSV file: {filepath}")
            file_content = convert_csv_to_text(filepath)
            if file_content is None:
                print(f"Skipping file {filename} due to CSV processing error.")
                continue
        elif filename.lower().endswith(".txt"):
            print(f"Processing text file: {filepath}")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    file_content = f.read()
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
        else:
            continue

        extracted_result = extract_data(file_content)
        if extracted_result is not None:
            output_filename = os.path.splitext(filename)[0] + f"_ida_{index}.txt"
            output_path = os.path.join(output_folder, output_filename)
            try:
                with open(output_path, "w", encoding="utf-8") as out_f:
                    out_f.write(extracted_result)
                print(f"Extracted data saved to: {output_path}")
            except Exception as e:
                print(f"Error writing to {output_filename}: {e}")
        else:
            print(f"Skipping file {filename} due to extraction error.")

if __name__ == "__main__":
    process_files(input_folder, output_folder)
