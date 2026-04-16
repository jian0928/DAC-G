import argparse
import json
import logging
import os
import re
import networkx as nx
from knockknock import wechat_sender
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
from utils import get_raw_dataset

STOP_WORDS = set(stopwords.words('english'))
PUNCTUATIONS = set("!\"#$%&'()*+-./:;<=>?@[\]^_`{|}~")

def split_short_sentences(text):
    parts = re.split(r'([?,;.!])', text)
    sentences = []
    if len(parts) <= 1:
        if len(text.strip()) > 2:
            return [text.strip()]
        else:
            return []
    for i in range(0, len(parts) - 1, 2):
        sentence = (parts[i] + parts[i + 1]).strip()
        if len(sentence) > 2:
            sentences.append(sentence)
    if parts[-1].strip() and len(parts[-1].strip()) > 2:
        sentences.append(parts[-1].strip())
    final_sentences = []
    for s in sentences:
        cleaned = re.sub(r'^[?,;.!]+', '', s).strip()
        if len(cleaned) > 1:
            final_sentences.append(s)
    return final_sentences

def preprocess_sentence(sentence):
    sentence = sentence.lower()
    words = re.findall(r'\b\w+\b', sentence)
    filtered_words = [
        word for word in words
        if word not in STOP_WORDS
           and word not in PUNCTUATIONS
           and not word.isdigit()
    ]
    return " ".join(filtered_words)

def calculate_similarity_matrix(sentences):
    processed_sentences = [preprocess_sentence(s) for s in sentences]
    if not any(processed_sentences):
        return np.zeros((len(sentences), len(sentences)))
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix
    except ValueError:
        return np.zeros((len(sentences), len(sentences)))

def extract_short_sentences(
        question,
        Qid,
        num_top_sentences,
        theta,
        damping_factor,
        epsilon
):
    full_text = question
    sentences = split_short_sentences(full_text)
    if not sentences:
        return [full_text]
    num_nodes = len(sentences)
    if num_nodes <= num_top_sentences:
        return sentences
    similarity_matrix = calculate_similarity_matrix(sentences)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            similarity = similarity_matrix[i, j]
            if similarity >= theta:
                graph.add_edge(i, j, weight=similarity)
    if len(graph.edges) == 0:
        return sentences[:num_top_sentences]
    scores = {i: 1.0 for i in range(num_nodes)}
    max_iterations = 100
    for k in range(max_iterations):
        new_scores = {}
        converged = True
        for i in range(num_nodes):
            rank_sum = 0.0
            neighbors_of_i = list(graph.neighbors(i))
            if neighbors_of_i:
                for j in neighbors_of_i:
                    S_ij = graph[i][j]['weight']
                    neighbors_of_j = list(graph.neighbors(j))
                    sum_S_jk = sum(graph[j][k]['weight'] for k in neighbors_of_j)
                    if sum_S_jk > 0:
                        rank_sum += (S_ij / sum_S_jk) * scores[j]
            new_scores[i] = (1 - damping_factor) + damping_factor * rank_sum
        for i in range(num_nodes):
            if abs(new_scores[i] - scores[i]) >= epsilon:
                converged = False
                break
        scores = new_scores
        if converged:
            break
    scored_sentences = [(scores[i], s) for i, s in enumerate(sentences)]
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    summary_sentences = [s for score, s in scored_sentences[:num_top_sentences]]
    summary = sorted(summary_sentences, key=lambda s: sentences.index(s))
    return summary

def extract_key():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    script_args = parser.parse_args()

    data = get_raw_dataset(script_args.dataset_name, "ann")

    THETA = 
    DAMPING = 
    EPSILON = 
    NUM_SENTENCES = 

    results_item = []
    for i, Q in tqdm(enumerate(data)):
        assistant_content = ""
        user_content = ""
        for item in Q["prompt"]:
            if item["role"] == "user":
                user_content = f"{user_content}{item['content']}"
            elif item["role"] == "assistant":
                assistant_content = f"{assistant_content}{item['content']}"

        assistant_content_A = assistant_content[:] + Q["response_A"]["content"]
        assistant_content_B = assistant_content + Q["response_B"]["content"]

        wrap_data = [user_content, assistant_content_A, assistant_content_B]

        results = []
        for content in wrap_data:
            summary_sequence = extract_short_sentences(
                content,
                Q['id'],
                NUM_SENTENCES,
                THETA,
                DAMPING,
                EPSILON
            )
            results.append(summary_sequence)
        result_item = {
            "id": Q['id'],
            "prompt": str("".join(results[0])),
            "response_A": str("".join(results[1])),
            "response_B": str("".join(results[2])),
        }
        results_item.append(result_item)

    output_filepath = f"./results/{script_args.dataset_name}/{script_args.dataset_name}_extract_key.json"
    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(results_item, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    extract_key()