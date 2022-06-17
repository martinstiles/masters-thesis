"""
Helper methods
"""

def cosine_similarity(X, Y):
    assert len(X) == len(Y), "Dimension of vectors does not match"

    dot_product = 0
    X_magnitude = 0
    Y_magnitude = 0
    for i in range(len(X)):
        X_i = X[i]
        Y_i = Y[i]
        dot_product += X_i * Y_i
        X_magnitude += X_i ** 2
        Y_magnitude += Y_i ** 2
    
    return dot_product / ( X_magnitude * Y_magnitude )



"""
FOR TESTING PURPOSES:
"""

from pathlib import Path
from nltk.tokenize import sent_tokenize
from functools import reduce

FULL_DATA_PATH = Path(__file__).parent.parent / "data" / "will_smith_dataset.txt" 
SMALL_DATA_PATH = Path(__file__).parent.parent / "data" / "will_smith_dataset_small.txt" 
TEST_DATA_PATH = Path(__file__).parent.parent / "data" / "will_smith_dataset_cleaned_small.txt"

def prepare_data(dataset):
    """Performs necessary cleaning of dataset and returns a list of the sentences in the corpus

    Args:
        dataset (str): a string to specify which dataset should be used
    
    Returns:
        prepared_sentences (list(str)): A list containing each sentence in the corpus as a string
    """
    datapath_dict = {
        "full": FULL_DATA_PATH,
        "small": SMALL_DATA_PATH
    }
    data_path = datapath_dict[dataset]
    
    with open(data_path, "r") as file:
        articles = file.readlines()
    
    # TODO: divide into smaller functions
    # Remove "small" sentences -> noice like "Recommended Stories", "You Might Also Like" or "NBC"
    cleaned_articles = []
    for article in articles:
        long_sentences = [sentence for sentence in article.split("  ")
                          if len(sentence.split(" ")) > 7]
        # remove last sentence if it ends with "..."
        cleaned_articles.append(long_sentences)  # TODO: LEAD? [1:5]
    
    # Extract multiple sentences that are disguised as one (after split on "  ")
    all_sentences = []
    for article in cleaned_articles:
        for potential_sentence in article:
            sentences = sent_tokenize(potential_sentence)
            for sentence in sentences:
                if len(sentence.split(" ")) < 50:  # TODO: max length?
                    all_sentences.append(sentence)

    all_sentences = [sentence for sentence in all_sentences
                     if sentence[-3:] != "..."]
    
    objects = []
    for sentence in all_sentences:
        sentence_object = {
            "sentence": sentence,
            "pos_in_article": 1,
            "embedding": None,
            "tfidf_score": None,
            "distances": []
        }
        objects.append(sentence_object)
    
    return objects


def load_test_data(dataset):
    assert dataset in ["test", "full", "small"], "Data type specified is incorrect. Try \"full\", \"small\" or \"test\""
    
    if dataset == "test":
        with open(TEST_DATA_PATH, "r") as file:
            articles = file.readlines()
            sentences_2D = [article.split(" | ") for article in articles]
            # Flatten sentences into 1D list
            sentences = reduce(lambda a, b: a + b, sentences_2D)
        return sentences
    
    sentences = prepare_data(dataset)
    return sentences