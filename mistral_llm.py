import requests
import json
import pandas as pd

# Define the API endpoint
endpoint = 'Add endpoint URL here'

# Define the prompt for the task
llm_prompt = '''
```Our task here is to label the given text samples into a political orientation of either 0 (left-leaning) or 1 (right-leaning). DO NOT PROVIDE ANY ADDITIONAL EXPLANATION, ONLY OUTPUT THE NUMERICAL LABEL (0 or 1). Here's a couple examples explaining the labels: 
Example 1(Label: 0):
I know the hon. Gentleman is giving a speech about a popular view of the private finance initiative, but I wish to make him aware of the Atkinson Morley wing at St George's Hospital, which is a brilliant neurological centre that cost £50 million through PFI. It was built in the late 1990s, and it has saved hundreds and thousands of people. It is a building, and an opportunity to have a service, that was not coming any other way. I give thanks for that PFI deal, and I give thanks for those people who have been saved by it.
Example 2(Label: 1):
"Yes, my Lords, I think your Lordships' Constitution Committee makes outstanding contributions to all thinking on constitutional matters. As I indicated in my previous answer, we are seeking approaches to always create good relations—as far as we can—between the different Administrations of these islands. That means good will, and every party has to show that good will.
My Lords, I have indicated previously to your Lordships' House that the Government are determined to take the various aspects of constitutional consideration forward; I gave the House examples of the different workstreams. I simply do not agree with the noble Lord that there is not cross-party agreement on certain things. For example, the removal of the Fixed-term Parliaments Act was agreed across the House and the principle of it was subject to very extensive consultation and examination." ```
'''

file_path = 'orientation-gb-train.tsv'

# Load the data from the TSV file
dataframe = pd.read_csv(file_path, sep='\t')

updated_data = []

# Iterate over the rows of the dataframe
for idx, row in dataframe[['id', 'text']].iterrows():
    speech_id = row['id']
    speech_text = row['text']
    prompt = f"[INST] {llm_prompt} Label This now, and output should just be the Label: 0 or 1 ;  'text': '{speech_text}' [/INST]"

    # Make an API call with the prompt and other parameters
    res = requests.post(endpoint, json={
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "max_tokens": 4000,
        "prompt": prompt,
        "request_type": "language-model-inference",
        "temperature": 0.0,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": [
            "[/INST]",
            "</s>"
        ],
        "repetitive_penalty": 1
    }, headers={
        "Authorization": "API Key Here",
    })

    # Process the API response
    output = res.json()['choices'][0]['message']['content']

    new_obj = {
        'id': speech_id,
        'Label': output,
    }

    updated_data.append(new_obj)

# Write the updated data back to the JSON file
with open('label_output.json', 'a') as file:
    json.dump(updated_data, file, indent=4)