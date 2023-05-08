import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(base_folder):
    """
    Loads the preprocessed data from JSON files into a nested dictionary
    """
    data = {}

    for problem_folder in os.listdir(base_folder):
        problem_path = os.path.join(base_folder, problem_folder)
        data[problem_folder] = {}
        for file in os.listdir(problem_path):
            if file.endswith(".json"):
                with open(os.path.join(problem_path, file), 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    candidate_id = file.split('.')[0]
                    text = ' '.join(content.keys())
                    data[problem_folder][candidate_id] = text

        # Load unknown files
        unknown_path = os.path.join(problem_path, 'unknown')
        if os.path.exists(unknown_path):
            data[problem_folder]['unknown'] = {}
            for file in os.listdir(unknown_path):
                if file.endswith(".json"):
                    with open(os.path.join(unknown_path, file), 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        unknown_id = file.split('.')[0]
                        text = ' '.join(content.keys())
                        data[problem_folder]['unknown'][unknown_id] = text

    return data


def save_tfidf(data, output_folder):
    """
    Extracts TF-IDF features and saves them as separate CSV files for each candidate and unknown
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for problem_id, problem_data in data.items():
        unknown_data = problem_data.pop('unknown', None)

        problem_df = pd.DataFrame(list(problem_data.items()), columns=["candidate_id", "text"])
        vectorizer = TfidfVectorizer()
        X_tfidf = vectorizer.fit_transform(problem_df["text"])

        feature_names = vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=feature_names, index=problem_df["candidate_id"])

        problem_output_folder = os.path.join(output_folder, problem_id)
        if not os.path.exists(problem_output_folder):
            os.makedirs(problem_output_folder)

        for candidate_id in tfidf_df.index:
            candidate_tfidf = tfidf_df.loc[candidate_id]
            candidate_tfidf.to_csv(os.path.join(problem_output_folder, f"{candidate_id}_tfidf.csv"))

        # Save unknown TF-IDF features
        if unknown_data:
            unknown_df = pd.DataFrame(list(unknown_data.items()), columns=["unknown_id", "text"])
            X_unknown_tfidf = vectorizer.transform(unknown_df["text"])
            tfidf_unknown_df = pd.DataFrame(X_unknown_tfidf.toarray(), columns=feature_names, index=unknown_df["unknown_id"])

            unknown_output_folder = os.path.join(problem_output_folder, 'unknown')
            if not os.path.exists(unknown_output_folder):
                os.makedirs(unknown_output_folder)

            for unknown_id in tfidf_unknown_df.index:
                unknown_tfidf = tfidf_unknown_df.loc[unknown_id]
                unknown_tfidf.to_csv(os.path.join(unknown_output_folder, f"{unknown_id}_tfidf.csv"))


if __name__ == "__main__":
    base_folder = r'C:\Users\ltrbn\OneDrive\Desktop\CMPE493\TP\preprocessed_test'
    output_folder = r'C:\Users\ltrbn\OneDrive\Desktop\CMPE493\TP\tfidf_features_test'
    data = load_data(base_folder)

    save_tfidf(data, output_folder)
