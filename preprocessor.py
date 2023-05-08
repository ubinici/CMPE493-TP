import os
import json
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from math import log

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def text_processor(text, n_grams='all'):
    """
    Process and tokenize the input text by removing punctuation, stopwords, and non-alphanumeric characters,
    and lemmatizing the words.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token.isalnum() and not token.isdigit()]

    if n_grams == 'all':
        bigrams = generate_bigrams(tokens)
        trigrams = generate_trigrams(tokens)
        return tokens + bigrams + trigrams
    elif n_grams == 'bigrams':
        return generate_bigrams(tokens)
    elif n_grams == 'trigrams':
        return generate_trigrams(tokens)
    else:
        return tokens


def generate_bigrams(tokens):
    bigrams = list(nltk.bigrams(tokens))
    return [f"{bigram[0]}_{bigram[1]}" for bigram in bigrams]


def generate_trigrams(tokens):
    trigrams = list(nltk.trigrams(tokens))
    return [f"{trigram[0]}_{trigram[1]}_{trigram[2]}" for trigram in trigrams]


def json_writer(problem_folder, candidate_folder, preprocessed_data, output_folder):
    """
    Write the preprocessed data for each candidate to a JSON file.
    """
    if problem_folder == "unknown":
        problem_output_folder = output_folder
    else:
        problem_output_folder = os.path.join(output_folder, problem_folder)
    os.makedirs(problem_output_folder, exist_ok=True)
    output_file_path = os.path.join(problem_output_folder, f"{candidate_folder}.json")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessed_data, f, ensure_ascii=False, indent=4)


def process_candidates(candidate_path, n_grams='all'):
    """
    Process all text files for a candidate and return a corpus of preprocessed tokens and bigrams.
    """
    candidate_corpus = []

    for txt_file in os.listdir(candidate_path):
        if txt_file.endswith('.txt'):
            with open(os.path.join(candidate_path, txt_file), 'r', encoding='utf-8') as f:
                text = f.read()
                lemmatized_text = text_processor(text, n_grams=n_grams)
                candidate_corpus.extend(lemmatized_text)

    return candidate_corpus


def process_unknown(unknown_path, problem_folder, output_folder, n_grams='all'):
    """
    Process all text files in the unknown folder and store the preprocessed data in a JSON file.
    """
    output_folder = os.path.join(output_folder, problem_folder, "unknown")
    os.makedirs(output_folder, exist_ok=True)

    for txt_file in os.listdir(unknown_path):
        if txt_file.endswith('.txt'):
            with open(os.path.join(unknown_path, txt_file), 'r', encoding='utf-8') as f:
                text = f.read()
                lemmatized_text = text_processor(text, n_grams=n_grams)
                log_frequencies = calculate_log_frequencies(lemmatized_text)
                json_writer("unknown", txt_file[:-4], log_frequencies, output_folder)


def calculate_log_frequencies(corpus):
    """
    Calculate the log frequency of each word or bigram in the corpus.
    """
    word_frequencies = {}
    for token in corpus:
        if token in word_frequencies:
            word_frequencies[token] += 1
        else:
            word_frequencies[token] = 1

    log_frequencies = {word: log(freq) for word, freq in word_frequencies.items()}

    return log_frequencies


def file_processor(folder_paths, output_folder):
    """
    Process all candidate folders and unknown folders in the given folder paths and store preprocessed data in a nested dictionary.
    """
    preprocessed_data = {}

    for folder_path in folder_paths:
        problem_folder = os.path.basename(folder_path)
        preprocessed_data[problem_folder] = {}

        for candidate_folder in os.listdir(folder_path):
            if candidate_folder.startswith('candidate'):
                candidate_path = os.path.join(folder_path, candidate_folder)
                candidate_corpus = process_candidates(candidate_path)
                log_frequencies = calculate_log_frequencies(candidate_corpus)
                preprocessed_data[problem_folder][candidate_folder] = log_frequencies
                json_writer(problem_folder, candidate_folder, log_frequencies, output_folder)
            elif candidate_folder == 'unknown':
                unknown_path = os.path.join(folder_path, candidate_folder)
                process_unknown(unknown_path, problem_folder, output_folder)

    return preprocessed_data


if __name__ == "__main__":
    base_folder = r'C:\Users\ltrbn\OneDrive\Desktop\CMPE493\TP\pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23\pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23\problem-test'
    folders_to_process = ['problem00001', 'problem00003', 'problem00005']
    output_folder = r'C:\Users\ltrbn\OneDrive\Desktop\CMPE493\TP\preprocessed_test'
    os.makedirs(output_folder, exist_ok=True)

    all_preprocessed_data = file_processor([os.path.join(base_folder, folder) for folder in folders_to_process], output_folder)
    