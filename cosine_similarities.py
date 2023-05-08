import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def cosine_sim(base_folder):
    data={}
    for problem_folder in os.listdir(base_folder):
        problem_path = os.path.join(base_folder, problem_folder)
        data[problem_folder]={}
        for file in os.listdir(problem_path):
            if file.endswith(".csv"):
                with open(os.path.join(problem_path, file), 'r', encoding='utf-8') as f:
                    content = pd.read_csv(f)
                    candidate_id=file.split(".")[0]
                    data[problem_folder][candidate_id]=content

        unknown_path = os.path.join(problem_path, 'unknown')
        if os.path.exists(unknown_path):
            data[problem_folder]['unknown'] = {}
            for file in os.listdir(unknown_path):
                if file.endswith(".csv"):
                    with open(os.path.join(unknown_path, file), 'r', encoding='utf-8') as f:
                        content = pd.read_csv(f)
                        unknown_id = file.split(".")[0]
                        data[problem_folder]['unknown'][unknown_id] = content

    return data


def normalize_list(lst):
    min_val = min(lst)
    max_val = max(lst)
    normalized_lst = [(x - min_val) / (max_val - min_val) for x in lst]
    return normalized_lst


def save_similarity(data,output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    score_data = {}
    for problem_folder in data:
        ground_truth_path=os.path.join(r'C:\Users\ltrbn\OneDrive\Desktop\CMPE493\TP\pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23\pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23\problem-test',problem_folder,"ground-truth.json")
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth=json.load(f)
        for counter, unknown in enumerate(data[problem_folder]['unknown']):
            candidate_scores = []
            for candidate in data[problem_folder]:
                if candidate != 'unknown':
                    unknown_vector = list(data[problem_folder]['unknown'][unknown].iloc[:, 1])
                    candidate_vector = list(data[problem_folder][candidate].iloc[:, 1])
                    unknown_array = np.array(unknown_vector)
                    candidate_array = np.array(candidate_vector)
                    unknown_array = unknown_array.reshape(1, -1)
                    candidate_array = candidate_array.reshape(1, -1)
                    similarity_score = cosine_similarity(unknown_array, candidate_array)
                    candidate_scores.append(similarity_score[0, 0])
            candidate_scores=normalize_list(candidate_scores)
            candidate_scores.append(ground_truth["ground_truth"][counter]["true-author"])
            score_data[unknown] = candidate_scores
        column_names = {0: "candidate00001", 1: "candidate00002", 2: "candidate00003", 3: "candidate00004",
                        4: "candidate00005", 5: "candidate00006", 6: "candidate00007", 7: "candidate00008",
                        8: "candidate00009"}
        df = pd.DataFrame(score_data).transpose()
        df = df.rename(columns=column_names)
        df.to_csv(os.path.join(output_folder, f"{problem_folder}_cosine_similarity.csv"))


if __name__ == "__main__":
    base_folder=r"C:\Users\ltrbn\OneDrive\Desktop\CMPE493\TP\tfidf_features_test"
    output_folder = r"C:\Users\ltrbn\OneDrive\Desktop\CMPE493\TP\cosine_similarities"
    data=cosine_sim(base_folder)
    save_similarity(data,output_folder)