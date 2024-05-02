# CSC 483 Final Project - Power and Ideology Detection Task CLEF 2024

## Project Overview
This project was part of the "Ideology and Power Identification in Parliamentary Debates 2024" task at the CLEF 2024 conference. The goal was to analyze parliamentary speeches and identify the ideology of the speaker's party based on the content of the speeches. We focused our efforts on the British parliamentary debate corpus, which consisted of over 22,000 speeches.

## Contributors
* Madison Ryan
* Abhyuday Singh

## Files
1. clef-byteocrats.ipynb: This Jupyter Notebook file is responsible for loading the dataset, applying preprocessing techniques (such as stopword removal, punctuation removal, tokenization, and lemmatization), and running various machine learning models with evaluation metrics.
2. mistral_llm.py: This Python script utilizes a large language model (LLM) - Mistral 8x7b to classify the parliamentary speeches. It makes API calls to the LLM, processes the input data, and writes the output predictions to a text file

## How to Run
### 1. Data Preprocessing and Machine Learning Models:
* Open the clef-byteocrats.ipynb file in a Jupyter Notebook environment.
* Run the cells to load the dataset, preprocess the data, and train/evaluate the machine learning models.
* The notebook will output the performance metrics for the various models.
  
### 2. Large Language Model Classification
* Ensure you have the necessary API credentials to access the LLM.
* Update the API endpoint and credentials in the mistral_llm.py file.
* Run the script, and the output predictions will be saved in a text file
