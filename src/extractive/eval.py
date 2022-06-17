from pathlib import Path
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from datasets import load_dataset
from main import summarize

CNN_DAILYMAIL_PATH = Path(__file__).parent.parent.parent / "data" / "cnn_dailymail" / "test.csv"


def load_data(start_index, num_articles):
    data = load_dataset('cnn_dailymail', '3.0.0')
    test_data = data["test"][start_index:start_index+num_articles]

    summary_objects = []
    for article, highlights in zip(test_data["article"], test_data["highlights"]):
        sentences = sent_tokenize(article)
        sentence_objects = []
        for sentence in sentences:
            # if len(sentence.split(" ")) < 41:
            sentence_objects.append({
                "sentence": sentence,
                "pos_in_article": 0,  # Pos in article is not necessary (single document)
                "embedding": None,
                "tfidf_score": None,
                "distances": []
            })
        summary_objects.append(
            {
                "sentence_objects": sentence_objects,
                "summary": highlights
            }
        )

    return summary_objects


""" LOAD DATA AND SUMMARIZER """
start_index = 11458  # SHOULD CORRESPOND WITH THE LAST ID IN eval.txt (+1)
num_articles = 32
data = load_data(start_index, num_articles)


max_sentence_length = 0  # Possible hyperparameter
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
counter = start_index
for article_object in data:
    try:
        """ GENERATE SUMMARY """
        summary = summarize(
            sentence_objects = article_object["sentence_objects"],
            num_sentences = 2
        )
        article_object["generated_summary"] = summary


        """ GET SCORES """
        measure_to_index_map = {
            "precision": 0,
            "recall": 1,
            "f1": 2
        }
        scores = scorer.score(article_object["summary"], article_object["generated_summary"])
        rouge1_precision = (scores["rouge1"][measure_to_index_map["precision"]])
        rouge1_recall = (scores["rouge1"][measure_to_index_map["recall"]])
        rouge1_f1 = (scores["rouge1"][measure_to_index_map["f1"]])
        rouge2_precision = (scores["rouge2"][measure_to_index_map["precision"]])
        rouge2_recall = (scores["rouge2"][measure_to_index_map["recall"]])
        rouge2_f1 = (scores["rouge2"][measure_to_index_map["f1"]])
        rougeL_precision = (scores["rougeL"][measure_to_index_map["precision"]])
        rougeL_recall = (scores["rougeL"][measure_to_index_map["recall"]])
        rougeL_f1 = (scores["rougeL"][measure_to_index_map["f1"]])


        """ Save to file """
        with open(Path(__file__).parent / "eval.txt", "a") as file:
            file.write(f"{counter};{rouge1_precision};{rouge1_recall};{rouge1_f1};{rouge2_precision};{rouge2_recall};{rouge2_f1};{rougeL_precision};{rougeL_recall};{rougeL_f1}\n")

        print(f"Status: {counter}/{start_index + num_articles - 1}")
        counter += 1

    except Exception:
        print("Something went wrong with article:", counter)
        with open(Path(__file__).parent / "eval_article_errors.txt", "a") as file:
            file.write(str(counter) + "\n")
        counter += 1
