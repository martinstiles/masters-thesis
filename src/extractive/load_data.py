"""
Methods for loading and preprocessing the dataset
"""

from pandas import read_csv
from pathlib import Path
from nltk.tokenize import sent_tokenize
import re

ARTICLES_PATH = Path(__file__).parent.parent.parent / "data" / "clusters_large.txt"
ENTITIES_PATH = Path(__file__).parent.parent.parent / "data" / "entities_large.txt"


def remove_duplicate_articles(df_articles):
    # An article is regarded as a duplicate if it has the same title
    titles = []
    article_rows = []
    for _, row in df_articles.iterrows():
        title = row["title"]
        if title not in titles:
            titles.append(title)
            article_rows.append(row)
    
    return article_rows


# def preprocess(articles):
#     """ Performs necessary preprocessing before sentence tokenization can be performed
#         Mostly deals with "nn", which is a code for whitespace in the dataset.

#     Args:
#         articles (list[str]): A list of complete articles

#     Returns:
#         articles: _description_
#     """
#     articles = [article.replace('“', '"') for article in articles]
#     articles = [article.replace('”', '"') for article in articles]
#     articles = [article.replace(".nn", ". ") for article in articles]
#     articles = [article.replace('"nn', '" ') for article in articles]
#     articles = [article.replace('nn"', ' "') for article in articles]
    
#     # Remove "nn" followed by a big letter
#     for i in range(len(articles)):
#         matches = re.findall('[n]{2}[A-Z]', articles[i])
#         for match in matches:
#             big_letter = match[2]
#             articles[i] = articles[i].replace(match, " " + big_letter)
    
#     return articles


def get_offset_difference_between_body_and_fulltext(article_rows, event_id):
    """ Generates the offset to the body of the article. Offset originally points to an index in
        "fulltext", but we only use the sentences in "body", and thus we must find the difference
        in offset between them.

    Args:
        article_rows (list[dict]): a list of article objects from the pandas df
        event_id (int): current article index

    Returns:
        offset_before_body (int): the difference in offsets between body and fulltext
    """
    title = article_rows[event_id]["title"]
    summary = article_rows[event_id]["summary"]
    offset_before_body = len(title) + len(summary)
    if len(title) != 0: offset_before_body += 2
    if len(summary) != 0: offset_before_body += 2
    return offset_before_body


def get_sentences_and_offsets(article_sentences, offset_difference):
    """ Creates a list of objects containing each sentence and two offsets: one for the
        beginning of the sentence (from) and one for the end of the sentence (to). This can be
        used to check if an entity offset is between the sentence offsets.

    Args:
        article_sentences (list[str]): Every sentence in an article after split on double space
        offset_difference (int): the difference in offsets between body and fulltext

    Returns:
        list[dict]: a list with sentence objects (see below for object keys)
    """
    # Create a list of objects containing each sentence and the offset to the first character in the sentence
    sentences_and_offsets = []
    previous_offset = offset_difference  # Begins at offset for body (after title and summary)
    is_first_sentence = True
    for sentence in article_sentences:
        sentence_offset = 2 + previous_offset + len(sentence)

        # Should not do + 2 on first sentence -> + 2 comes from the spaces that articles are split on
        if is_first_sentence:
            sentence_offset -= 2
            is_first_sentence = False
        
        sentences_and_offsets.append(
            {
                "sentence": sentence,
                "from": previous_offset,
                "to": sentence_offset
            }
        )
        previous_offset = sentence_offset

    return sentences_and_offsets


def split_into_sentences_and_mark_with_entities(article_rows, df_entities):
    """ Each article will be devided into sentences, first by splitting on double space and then
        by applying nltk's sentence_tokenizer. However, before the tokenizer is applied, every
        sentence is marked with entities by using the offset from the entity dataset.

    Args:
        article_rows (list[dict]): Article objects from the pandas df
        df_entities (dataframe): Pandas dataframe with every entity in the cluster

    Returns:
        list[list[dict]]: Every marked sentence in every article on the form
            [[
                {
                    "sentence": sentence,
                    "entities": entities_in_sentence
                }
            ]]
    """
    articles = [row["body"] for row in article_rows]
    sentences_2D = [article.split("  ") for article in articles]

    # TODO: Mark sentenes with entities
    marked_sentences_2D = []
    for event_id in range(len(articles)):  # an event corresponds to an article
        article_sentences = sentences_2D[event_id]
        # offset_difference says how much we have to subtract from fulltext offset to get body offset
        offset_difference = get_offset_difference_between_body_and_fulltext(article_rows, event_id)
        sentences_and_offsets = get_sentences_and_offsets(article_sentences, offset_difference)

        entities_in_article = df_entities.loc[df_entities["eventId"] == event_id].sort_values(by = "offset")

        marked_sentences = []
        for obj in sentences_and_offsets:
            sentence, sentence_start_offset, sentence_end_offset = obj["sentence"], obj["from"], obj["to"]
            entities_in_sentence = set()

            for _, row in entities_in_article.iterrows():
                entity_offset, wiki_data_id = row["offset"], row["wikidataId"]
                if sentence_start_offset <= entity_offset < sentence_end_offset:
                    entities_in_sentence.add(wiki_data_id)
                
            marked_sentences.append(
                {
                    "sentence": sentence,
                    "entities": entities_in_sentence
                }
            )
        marked_sentences_2D.append(marked_sentences)

    # Use sent_tokenizer on sentences, because the split on double space is not perfect
    new_marked_sentences_2D = []
    for marked_sentences in marked_sentences_2D:
        new_marked_sentences = []
        for marked_sentence in marked_sentences:
            sentence, entities = marked_sentence["sentence"], marked_sentence["entities"]
            new_sentences = sent_tokenize(sentence)

            for new_sentence in new_sentences:
                new_marked_sentences.append(
                    {
                        "sentence": new_sentence,
                        "entities": entities
                    }
                )
        new_marked_sentences_2D.append(new_marked_sentences)
        
    # Add punctuation at end of sentence if there is none
    for i in range(len(new_marked_sentences_2D)):
        for j in range(len(new_marked_sentences_2D[i])):
            if new_marked_sentences_2D[i][j]["sentence"][-1] not in [".", "!", "?"]:
                new_marked_sentences_2D[i][j]["sentence"] = new_marked_sentences_2D[i][j]["sentence"] + "."

    return new_marked_sentences_2D


def is_valid_sentence(sentence, max_sentence_length = 50):
    """ Checks if a sentence should be included

    Args:
        sentence (str): a string with the sentence that should be checked
        no_long_sentences (bool): True if there should be low upper limit on sentence length

    Returns:
        boolean: False if number of words is outside the bounds or if the sentence contains
            some of the terms listed. True otherwise.
    """
    num_words = len(sentence.split(" "))

    if not (num_words > 8 and num_words < max_sentence_length):
        return False

    black_listed_terms = [
        "contact me",
        "play now",
        "for free",
        "getty images",
        "keep up with",
        "[+]"
    ]
    for term in black_listed_terms:
        if term in sentence.lower():
            return False
    
    if sentence[-3:] == "...":
        return False

    return True


def load_data(cluster_id, use_lead = False, max_sentence_length = 50, use_small_dataset = False):
    """ Loads sentences from dataset, marked with entity IDs of entities in the sentences ???? TODO
    
    Args:
        no_long_sentences (bool): True if there should be low upper limit on sentence length TODO

    Returns:
        sentence_objects (list[dict]): Every sentence in the corpus as objects, on the form:
            [ {
                "sentence": str,
                "pos_in_article": float,
                "embedding": vector or None,
                "tfidf_score": float or None,
                "distances": list[float]
                TODO
            } ]
    """
    # Load articles from file
    df_articles = read_csv(ARTICLES_PATH, sep=";")
    df_articles = df_articles.loc[df_articles["clusterId"] == cluster_id]
    df_articles = df_articles[["title", "summary", "body"]]

    # Load entities
    df_entities = read_csv(ENTITIES_PATH, sep=";")
    df_entities = df_entities.loc[df_entities["clusterId"] == cluster_id]

    # Gets the body of non-duplicate articles
    article_rows = remove_duplicate_articles(df_articles)

    all_marked_sentences_2D = split_into_sentences_and_mark_with_entities(article_rows, df_entities)


    # If use_lead is specified we only keep the first three sentences
    if use_lead:
        all_marked_sentences_2D = [sentences[:3] for sentences in all_marked_sentences_2D]

    # Create sentence objects
    all_sentence_objects = []
    for marked_sentences in all_marked_sentences_2D:
        num_sentences_in_article = len(marked_sentences)
        for i in range(num_sentences_in_article):
            sentence_object = {
                "sentence": marked_sentences[i]["sentence"],
                "entities": marked_sentences[i]["entities"],
                "pos_in_article": i / num_sentences_in_article,
                "embedding": None,
                "tfidf_score": None,
                "distances": []
            }
            all_sentence_objects.append(sentence_object)

    valid_sentence_objects = [obj for obj in all_sentence_objects
                        if is_valid_sentence(obj["sentence"], max_sentence_length)]

    return valid_sentence_objects


def filter_sentences(sentence_objects, entities):
    if len(entities) == 0:
        return sentence_objects

    filtered_sentence_objects = []
    for sentence_object in sentence_objects:
        entities_in_sentence = sentence_object["entities"]
        for entity in entities:
            if entity in entities_in_sentence:
                filtered_sentence_objects.append(sentence_object)
                break

    return filtered_sentence_objects


def filter_sentences_old(sentence_objects, entities):
    if len(entities) == 0:
        return sentence_objects

    filtered_sentence_objects = []
    for sentence_obj in sentence_objects:
        for entity in entities:
            if "" + entity.lower() + "" in sentence_obj["sentence"].lower():  # TODO: Entity with spaces around... Does not work with . for instance
                filtered_sentence_objects.append(sentence_obj)
                break

    return filtered_sentence_objects


"""
TESTING
"""

if __name__ == "__main__":
    data = load_data(0, use_small_dataset=True)
    print(len(data))
    entities = ["Q7747"]
    filtered_sentences = filter_sentences(data, entities)
    # filtered_sentences = filter_sentences2(data, ["putin"])
    print(len(filtered_sentences))
    for s in filtered_sentences:
        print(s["sentence"])
    