"""
Utility methods for the project
"""
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *

def add_embedding_to_sentence_objects(sentence_objects, encoder = "BERT_mini"):
    """ Creates a sentence embedding for each sentence in the corpus by using the specified encoder

    Args:
        sentence_objects (list[dict]): A list of sentence objects (see load_data() for type)
        encoder (str): A string to specify which encoder should be used
    
    Returns:
        list(vec): A list of vectors representing the original sentenes
    """
    # Check that given encoder is allowed
    assert encoder in ["BERT_mini"], "Given sentence encoder is not allowed."
    
    sentences = [obj["sentence"] for obj in sentence_objects]

    if encoder == "BERT_mini":
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        embeddings = model.encode(sentences)
        for i in range(len(embeddings)):
            sentence_objects[i]["embedding"] = embeddings[i]
        return embeddings


def add_cluster_id_to_sentence_objects(sentence_objects, DEBUG):
    """ Performs K-means clustering on sentence embeddings and adds the associated cluster ID
        to the sentence object. Max number of clusters cannot surpass the number of sentences.

    Args:
        sentence_objects (list[dict]): A list of sentence objects
        sentence_embeddings (list[vector]): A list of vectors which represents the sentence embeddings
    """
    sentence_embeddings = [obj["embedding"] for obj in sentence_objects]
    
    k_means_args = {
        "init": "random",
        "n_init": 5,
        "max_iter": 150,
        "random_state": 42
    }

    silhouette_scores = []
    min_k = 5
    # Number of clusters must be less than number of sentences - a check is performed in filter_sentences()
    max_k = min(12, len(sentence_objects) - 1)  # TODO: k > 12 seems to give bad results (manual inspection)?
    for k in range(min_k, max_k):
        k_means = KMeans(
            n_clusters = k,
            **k_means_args
        )    
        k_means.fit_predict(sentence_embeddings)
        score = silhouette_score(sentence_embeddings, k_means.labels_)
        silhouette_scores.append((k, score))

    # Get the k value with highest silhouette score
    best_k = max(silhouette_scores, key = lambda tup: tup[1])[0]

    if DEBUG:
        print("Best number of clusters in K-means:", best_k)

    best_k_means = KMeans(
        n_clusters = best_k,
        **k_means_args
    )
    best_k_means.fit_predict(sentence_embeddings)

    # Add cluster ID to each sentence
    for i in range(len(sentence_objects)):
        cluster_id = best_k_means.labels_[i]
        sentence_objects[i]["cluster_id"] = cluster_id


def get_top_cluster_ids(sentence_objects, num_sentences = 3):
    """ Finds the top clusters according to how many sentences are in them.

    Args:
        sentence_objects (list[dict])
        num_sentences (int): The number of sentences to include in the extractive summary

    Returns:
        top_clusters (list[int]): A list of cluster IDs sorted by score, where
            the first element has the highest score.
    """
    cluster_ids = [obj["cluster_id"] for obj in sentence_objects]
    cluster_counts = Counter(cluster_ids)
    print("TOP CLUSTERS:")
    print(cluster_counts.most_common(num_sentences))
    top_cluster_ids = [cluster_id for cluster_id, count in cluster_counts.most_common(num_sentences)]
    return top_cluster_ids



def get_top_sentence_object_in_cluster(sentence_objects, DEBUG):
    # TODO: DocString - TextRank approximation
    if DEBUG:
        print("Num sentences in top cluster:", len(sentence_objects))
    
    for obj1 in sentence_objects:  # TODO: Can be improved from n**2 (to n**2/2...)
        for obj2 in sentence_objects:
            similarity = cosine_similarity(obj1["embedding"], obj2["embedding"])
            obj1["distances"].append(similarity)
    
    # Find average distance
    for obj in sentence_objects:
        obj["avg"] = sum(obj["distances"]) / len(obj["distances"])
    
    # Sort on avg
    sentence_objects.sort(key = lambda obj: obj["avg"], reverse = True)
    
    return sentence_objects[0]


def get_top_sentence_objects(all_sentence_objects, top_cluster_ids, DEBUG):
    """ Finds the top sentence for each cluster

    Args:
        sentence_objects (list[dict])
        top_cluster_ids (list[int]): A list of cluster IDs sorted by score, where
            the first element has the highest score.

    Returns:
        top_sentences (list[str]): A list of the top sentences
    """
    top_sentence_objects = []
    for top_cluster_id in top_cluster_ids:
        sentence_objects_in_top_cluster = [obj for obj in all_sentence_objects if obj["cluster_id"] == top_cluster_id]
        top_sentence_object = get_top_sentence_object_in_cluster(sentence_objects_in_top_cluster, DEBUG)
        top_sentence_objects.append(top_sentence_object)
    
    return top_sentence_objects


def combine_sentences(top_sentence_objects):
    top_sentence_objects.sort(key = lambda obj: obj["pos_in_article"])
    top_sentences = [obj["sentence"] for obj in top_sentence_objects]

    combined_sentences = ""
    for sentence in top_sentences:
        if sentence[-1] not in [".", "!", "?", '."']:
            combined_sentences += sentence + ". "
        else:
            combined_sentences += sentence + " "

    return combined_sentences
