# Author: Lau Tsz Yeung Anson
# Contact: s11327605@gm.cyut.edu.tw/tylau70242@gmail.com
# Updated_Date: 2025-07-09
import time
from ollama import Client
import pandas as pd
from pydantic import BaseModel
import json
from pathlib import Path
import base64

from dotenv import load_dotenv
import os
load_dotenv()

# JSON Schema
# You can add more key in this part, to fit different task.
class Result(BaseModel):
  response: str

# Get response from LLM
def LLM(client: Client, model: str, system_prompt: str, user_prompt: str, img: str = None):
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    
    # If an image path is provided, add it to the 'images' key of the user message.
    # The ollama library will handle reading the file from the path.
    if img:
        messages[1]['images'] = [img]
    
    response = client.chat(
        model=model,
        messages=messages,
        options={
            "seed": 42,
            "temperature": 0,
        },
        format=Result.model_json_schema(),
    )

    response_content = response['message']['content']
    return response_content

# Extract JSON from response
def extract_json(response: str):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return json.dumps({"Error": f"'{response}' is not a valid JSON"})

def main(client: Client, model: str):
    data = pd.read_csv("demo.csv")
    num_files = len(data)
    result = []
    start_time = time.time()

    for i in range(num_files):
        # Read Prompt (Use the same way if you have fixed user prompt)
        file_path = "prompt/system_prompt.md"
        with open(file_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        
        user_prompt = data['question'].iloc[i]
        
        img_path = None  # Initialize as None
        if 'img_path' in data.columns and pd.notna(data['img_path'].iloc[i]):
            # Directly get the file path from the CSV
            img_path = str(data['img_path'].iloc[i]).strip().strip('"').strip("'")
            
        print(f"Processing prompt: {user_prompt}")
        
        try:
            # Generate LLM response
            response = LLM(client, model, system_prompt, user_prompt, img_path)
            
            # Extract JSON from response
            json_response = extract_json(response)
            
        except Exception as e:
            # If an error occurs (e.g., model cannot handle images), create an error entry
            error_message = f"An error occurred for prompt '{user_prompt}': {e}"
            print(error_message)
            json_response = {"Error": error_message}

        result.append(json_response)
        print(json_response)

    df_result = pd.DataFrame(result)
    df_result.to_json('result.json', orient='records', indent=4)
    print("--- %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    # Change host and model here
    # GPU_Server_1, GPU_Server_2, GPU_Server_3
    client = Client(host=os.getenv("GPU_Server_1"))
    model = "qwen2.5vl:32b-q4_K_M"
    main(client, model)