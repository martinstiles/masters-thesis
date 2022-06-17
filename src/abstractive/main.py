# %%
from sklearn import cluster
from load_data import load_data, filter_sentences
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from methods import make_sentence_chunks, summarize_chunk, get_sentences_in_generated_summary, combine_summary_sentences
from time import time


def load_summarizer(model_type = "cnn_dailymail"):
    assert model_type in ["multi_news", "cnn_dailymail"], "Incorrect model type given."
    
    tokenizer = None
    model = None
    
    if model_type == "multi_news":
        tokenizer = AutoTokenizer.from_pretrained("google/pegasus-multi_news")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-multi_news")
    elif model_type == "cnn_dailymail":
        tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")
    
    summarizer = pipeline(task = "summarization", model = model, tokenizer = tokenizer)
    return summarizer

model_type = "multi_news"
# model_type = "cnn_dailymail"
summarizer = load_summarizer(model_type)

# %%

def summarize_recursively(
        sentences,
        summarizer,
        model_type,
        min_summary_length = 50,
        max_summary_length = 80,
        DEBUG = False
    ):
    summary_length = 9999  # high, random initial value
    final_summary = ""

    while summary_length > max_summary_length:
        sentence_chunks = make_sentence_chunks(sentences)
        sentences = []

        for chunk in sentence_chunks:
            summary = summarize_chunk(chunk, summarizer, min_summary_length, max_summary_length)
            new_sentences = get_sentences_in_generated_summary(
                summary = summary,
                model_type = model_type
            )
            sentences.extend(new_sentences)
        
        summary_length = 0
        for sentence in sentences:
            summary_length += len(sentence.split(" "))
        
        if summary_length <= max_summary_length:
            final_summary = combine_summary_sentences(sentences)
        
        print("")
        print("ITERATION DONE :)")
        print(summary_length)

    return final_summary

# %%

if __name__ == "__main__":
    cluster_id_to_entities = {
        0: ["Q37175"],  # Johnny Depp | Q229166 = Amber Heard
        1: ["Q317521"],  # Elon Musk | Q918 = Twitter --> TODO ??
        5: ["Q40096"],  # Will Smith | Q4109 = Chris Rock
        7: ["Q170572"],  # Alec Baldwin
        8: ["Q28967995"],  # Erling Braut Haaland
        10: ["Q355"],  # Facebook | Q36215 = Mark Zuckerberg
        12: ["Q22686"],  # Trump
    }

    cluster_id = 0
    entities = cluster_id_to_entities[cluster_id]
    # model_type = "cnn_dailymail"
    
    use_lead = len(entities) == 0
    sentences = load_data(
        cluster_id = cluster_id,
        use_lead = use_lead,
        max_sentence_length = 50
    )
    print("Total number:", len(sentences))
    filtered_sentences = filter_sentences(sentences, entities)
    print("Number of sentences to be summarized:", len(filtered_sentences))
    filtered_sentences = [ob["sentence"] for ob in filtered_sentences]
    
    # summarizer = load_summarizer(model_type)

    start = time()

    summary = summarize_recursively(
        filtered_sentences,
        summarizer,
        model_type,
        DEBUG = True
    )
    
    print("")
    print(summary)
    print("\nTotal time:", time() - start)
    print("\nCluster:", cluster_id)
    print("Model type:", model_type)

# %%
