"""
STATISTICS FOR THE CNN/DAILYMAIL TEST DATASET
"""
# %%
from datasets import load_dataset

dataset = load_dataset('cnn_dailymail', '3.0.0')
data = dataset["test"]

# %%

def print_summary_stats(data):
    print("Stats for summaries")
    summaries = [obj["highlights"] for obj in data]
    summary_lengths = [len(summary.split(" ")) for summary in summaries]
    # for i in range(len(summary_lengths)):
    #     if summary_lengths[i] == 535:
    #         print(data[i])
    print("Min:", min(summary_lengths))
    print("Max:", max(summary_lengths))
    print("avg:", sum(summary_lengths) / len(summary_lengths))
    summary_lengths.sort()
    print("Median:", summary_lengths[int(len(summary_lengths)/2)])

print_summary_stats(data)

# %%

def print_docuement_stats(data):
    print("Stats for documents")
    documents = [obj["article"] for obj in data]
    document_lengths = [len(document.split(" ")) for document in documents]
    print("Min:", min(document_lengths))
    print("Max:", max(document_lengths))
    print("avg:", sum(document_lengths) / len(document_lengths))
    document_lengths.sort()
    print("Median:", document_lengths[int(len(document_lengths)/2)])

print_docuement_stats(data) 

# %%
