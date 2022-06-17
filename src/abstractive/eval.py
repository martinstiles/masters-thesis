# %%
from pathlib import Path
from tracemalloc import start
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from datasets import load_dataset
from main import summarize_recursively, load_summarizer
from time import time

CNN_DAILYMAIL_PATH = Path(__file__).parent.parent.parent / "data" / "cnn_dailymail" / "test.csv"


def load_data(start_index, num_articles):
    data = load_dataset('cnn_dailymail', '3.0.0')
    test_data = data["test"][start_index:start_index+num_articles]

    summary_objects = []
    for article, highlights in zip(test_data["article"], test_data["highlights"]):
        sentences = sent_tokenize(article)

        summary_objects.append(
            {
                "sentences": sentences,
                "summary": highlights
            }
        )
    return summary_objects


""" SUMMARIZER """
model_type = "multi_news"
# model_type = "cnn_dailymail"
summarizer = load_summarizer(model_type = model_type)


# %%
""" LOAD DATA  """
filenames = {
    "cnn_dailymail": "eval_cnn.txt",
    "multi_news": "eval_mn.txt"
}

# mn mangler 3029, 3030
start_index = 3029  # SHOULD CORRESPOND WITH THE LAST ID IN eval.txt (+1)
num_articles = 2
data = load_data(start_index, num_articles)


scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
counter = start_index
for data_obj in data:
    try:
        """ GENERATE SUMMARY """
        summary = summarize_recursively(
            sentences = data_obj["sentences"],
            summarizer = summarizer,
            model_type = model_type,
            min_summary_length = 50,
            max_summary_length = 80,
            DEBUG = False
        )
        data_obj["generated_summary"] = summary


        """ GET SCORES """
        measure_to_index_map = {
            "precision": 0,
            "recall": 1,
            "f1": 2
        }
        scores = scorer.score(data_obj["summary"], data_obj["generated_summary"])
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
        with open(Path(__file__).parent / filenames[model_type], "a") as file:
            file.write(f"{counter};{rouge1_precision};{rouge1_recall};{rouge1_f1};{rouge2_precision};{rouge2_recall};{rouge2_f1};{rougeL_precision};{rougeL_recall};{rougeL_f1}\n")

        print(f"Status: {counter}/{start_index + num_articles - 1}")
        counter += 1
    
    except Exception:
        print("Something went wrong with article:", counter)
        with open(Path(__file__).parent / "eval_article_errors.txt", "a") as file:
            file.write(str(counter) + "\n")
        counter += 1


# %%
