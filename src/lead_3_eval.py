"""
CALCULATES THE AVERAGE F1-SCORE FOR LEAD-3 SENTENCES
"""

# %%
from pathlib import Path
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from datasets import load_dataset

CNN_DAILYMAIL_PATH = Path(__file__).parent.parent / "data" / "cnn_dailymail" / "test.csv"


def load_data():
    data = load_dataset('cnn_dailymail', '3.0.0')
    test_data = data["test"]

    summary_objects = []
    for article, highlights in zip(test_data["article"], test_data["highlights"]):
        # lead_3_sentences = ". ".join(article.split(". ")[:3])
        sentences = sent_tokenize(article)
        lead_3_sentences = " ".join(sentences[:3])
        summary_objects.append({
            "lead": lead_3_sentences,
            "summary": highlights
        })

    return summary_objects


# %%
data = load_data()

data[0]["summary"] = data[0]["summary"].replace("\n", " ")

# %%
print(data[0])

# %%
""" GET SCORES """
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

measure_to_index_map = {
    "precision": 0,
    "recall": 1,
    "f1": 2
}

c = 1
max_c = len(data)
for data_obj in data:
    scores = scorer.score(data_obj["summary"], data_obj["lead"])
    rouge1_scores.append(scores["rouge1"][measure_to_index_map["f1"]])
    rouge2_scores.append(scores["rouge2"][measure_to_index_map["f1"]])
    rougeL_scores.append(scores["rougeL"][measure_to_index_map["f1"]])
    if c % 1000 == 0:    
        print(f"{c} / {max_c}")
    c += 1


# %%
""" PRINT SCORES """
def get_avg(array):
    return sum(array) / len(array)

print("ROUGE-1 avg:", get_avg(rouge1_scores))
print("ROUGE-2 avg:", get_avg(rouge2_scores))
print("ROUGE-L avg:", get_avg(rougeL_scores))

# %%
