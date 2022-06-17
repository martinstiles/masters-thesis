"""
STATISTICS FOR THE MULTI-NEWS TEST DATASET
"""

# %%
from datasets import load_dataset

dataset = load_dataset("multi_news")
data = dataset["test"]

# %%

def print_summary_stats(data):
    print("Stats for summaries")
    summaries = [obj["summary"] for obj in data]
    summary_lengths = [len(summary.split(" ")) for summary in summaries]
    print("Min:", min(summary_lengths))
    print("Max:", max(summary_lengths))
    print("avg:", sum(summary_lengths) / len(summary_lengths))
    summary_lengths.sort()
    print("Median:", summary_lengths[int(len(summary_lengths)/2)])

print_summary_stats(data)


# %%

def print_collection_stats(data):
    print("Stats for collections")
    collections = [obj["document"] for obj in data]
    collection_sizes = [len(collection.split("|||||")) for collection in collections]
    print("Min:", min(collection_sizes))
    print("Max:", max(collection_sizes))
    print("avg:", sum(collection_sizes) / len(collection_sizes))
    collection_sizes.sort()
    print("Median:", collection_sizes[int(len(collection_sizes)/2)])

print_collection_stats(data) 

# %%

def print_docuement_stats(data):
    print("Stats for documents")
    collections = [obj["document"] for obj in data]
    documents = []
    for collection in collections:
        documents.extend(collection.split("|||||"))
        for doc in collection.split("|||||"):
            if len(doc.split(" ")) == 172393:
                with open("test.txt", "w") as file:
                    file.write(doc)
    document_lengths = [len(document.split(" ")) for document in documents]
    print("Min:", min(document_lengths))
    print("Max:", max(document_lengths))
    print("avg:", sum(document_lengths) / len(document_lengths))
    document_lengths.sort()
    print("Median:", document_lengths[int(len(document_lengths)/2)])

print_docuement_stats(data) 

# %%
