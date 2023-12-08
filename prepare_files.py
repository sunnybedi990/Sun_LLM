import pandas as pd
import os
# Load the datasets
import pandas as pd
import re
from bs4 import BeautifulSoup

def clean_text(text):
    # Remove HTML tags using BeautifulSoup
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text
def load_and_preprocess_data():
    preprocessed_file = 'preprocessed_qa_pairs.parquet'

    # Check if preprocessed file exists
    if os.path.exists(preprocessed_file):
        # Load the preprocessed data
        qa_pairs = pd.read_parquet(preprocessed_file)
    else:
        try:
            questions_df = pd.read_csv('Questions.csv', encoding='utf-8')
        except UnicodeDecodeError:
            questions_df = pd.read_csv('Questions.csv', encoding='latin1')  # or try 'cp1252' or 'ISO-8859-1'

        try:
            answers_df = pd.read_csv('Answers.csv', encoding='utf-8')
        except UnicodeDecodeError:
            answers_df = pd.read_csv('Answers.csv', encoding='latin1')  # or try 'cp1252' or 'ISO-8859-1'

            # Joining Questions and Answers
            qa_pairs = questions_df.merge(answers_df, left_on='Id', right_on='ParentId')
            qa_pairs = qa_pairs[['Title', 'Body_x', 'Body_y']]  # Title and Body from Questions, Body from Answers

            # Clean the text fields
            qa_pairs['Title'] = qa_pairs['Title'].apply(clean_text)
            qa_pairs['Body_x'] = qa_pairs['Body_x'].apply(clean_text)
            qa_pairs['Body_y'] = qa_pairs['Body_y'].apply(clean_text)

            # Combine title and body of questions to form the full question
            qa_pairs['Full_Question'] = qa_pairs['Title'] + ' ' + qa_pairs['Body_x']
            qa_pairs.to_parquet(preprocessed_file)
        # Create a list of tuples (question, answer)
    question_answer_pairs = list(zip(qa_pairs['Full_Question'], qa_pairs['Body_y']))

    return question_answer_pairs