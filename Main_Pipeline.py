import pandas as pd
import re
from Inference_wrapper import *

#Function that extracts the first number found in the text
def extract_number(text):
    """
    Extracts the first number found in the text.

    Args:
    text (str): The text to search.

    Returns:
    int or None: The extracted number or None if no number is found.
    """
    pattern = r'\d+'
    match = re.search(pattern, text)
    if match:
        number = int(match.group(0))
        return number
    else:
        return None
    
    # Function that detects which model is being use then puts the results in a new column 
def output_processing(file_path, output_path, model_flag=None):

    list_quantity = []
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Debug: Print the first few rows to inspect the content
    print("First few rows of the 'Results' column:")
    print(df['Results'].head())
    
    # Detect the model if not provided
    if model_flag is None:
        sample_text = df['Results'].iloc[0]
        if '<start_of_turn>model' in sample_text:
            model_flag = 'G'
        elif '|eot_id|><|start_header_id|>assistant<|end_header_id|' in sample_text:
            model_flag = 'M'
        else:
            raise ValueError("Unknown model type. Please specify 'M' for Meta or 'G' for Gemma.")
    
    # Process the file based on the detected or provided model
    if model_flag == 'M':
        # Meta model processing
        for text in df['Results']:
            # Ensure to correctly split using the specific pattern for META
            if '|eot_id|><|start_header_id|>assistant<|end_header_id|' in text:
                split_text = text.split('|eot_id|><|start_header_id|>assistant<|end_header_id|')[1]
                number = extract_number(split_text)
                list_quantity.append(number)
            else:
                # Handle the case where the pattern isn't found
                list_quantity.append(None)
    elif model_flag == 'G':
        # Gemma model processing
        for text in df['Results']:
            if '<start_of_turn>model' in text:
                split_text = text.split('<start_of_turn>model')[1]
                number = extract_number(split_text)
                list_quantity.append(number)
            else:
                # Handle the case where the pattern isn't found
                list_quantity.append(None)
    else:
        raise ValueError("Model flag should be 'M' for Meta or 'G' for Gemma.")
    
    # Add the list as a new column to the DataFrame
    df['Extracted_Number'] = list_quantity
    
    # Handle NaN values and convert to integers if it's Meta model
    df['Extracted_Number'] = df['Extracted_Number'].fillna(6)
    df['Extracted_Number'] = df['Extracted_Number'].astype(int)
    
    # Extract the original file name from the file_path
    original_file_name = file_path.split('/')[-1]
    
    # Save the transformed DataFrame back to a CSV file with the same name as the input file
    output_file = f"{output_path}/{original_file_name}"
    df.to_csv(output_file, index=False)
    
    print(f"Transformation complete. Output saved to: {output_file}")
    return df

system_progit clone https://github.com/ChrystyanPulidoLuna/ModelSift.git
mpt_zero_shot= f"""Read the tweet and classify it into one of the six emotion categories. sadness (0), joy (1), love (2), anger (3), fear (4), or surprise (5) Always answer the question in JSON format with the "emotion label" as a key and a number as a response.

"""
