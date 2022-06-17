"""
The main file from where the summarizer should be run
"""

from methods import *
from load_data import load_data, filter_sentences
from time import time


def summarize(sentence_objects, num_sentences = 3, DEBUG = False):
    
    # t0 = time.time()
    # sentence_time = time.time()
    add_embedding_to_sentence_objects(sentence_objects, "BERT_mini")
    # embedding_time = time.time()
    add_cluster_id_to_sentence_objects(sentence_objects, DEBUG)
    # cluster_time = time.time()

    # add_tfidf_score_to_sentence_objects(sentence_objects, no_long_sentences)

    top_cluster_ids = get_top_cluster_ids(sentence_objects, num_sentences)
    # top_cluster_time = time.time()
    top_sentence_objects = get_top_sentence_objects(sentence_objects, top_cluster_ids, DEBUG)
    # top_sentence_time = time.time()
    summary = combine_sentences(top_sentence_objects)
    print("Num words:", [len(sentence["sentence"].split(" ")) for sentence in top_sentence_objects])
    print("Num letters:", [len(sentence["sentence"]) for sentence in top_sentence_objects])
    
    return summary


if __name__ == "__main__":
    cluster_id_to_entities = {
        0: ["Q37175"],  # Johnny Depp | Q229166 = Amber Heard
        1: ["Q317521"],  # Elon Musk | Q918 = Twitter
        5: ["Q40096"],  # Will Smith | Q4109 = Chris Rock
        7: ["Q170572"],  # Alec Baldwin
        8: ["Q28967995"],  # Erling Braut Haaland
        10: ["Q355"],  # Facebook | Q36215 = Mark Zuckerberg
        12: ["Q22686"],  # Trump
    }
    cluster_id = 12
    # entities = []
    entities = cluster_id_to_entities[cluster_id]
    num_sentences = 3
    use_lead = False
    max_sentence_length = 35
    
    # LOAD DATA
    all_sentence_objects = load_data(cluster_id, use_lead, max_sentence_length)
    sentence_objects = filter_sentences(all_sentence_objects, entities)
    print("Number of sentences:", len(all_sentence_objects))
    print("Number of relevant sentences:", len(sentence_objects))

    # GENERATE SUMMARY
    start = time()
    summary = summarize(
        sentence_objects = sentence_objects,
        num_sentences = num_sentences,
        DEBUG = True
    )
    print("-------------------")
    print(summary)
    
    end = time()
    print("\nTotal time:", end - start)
