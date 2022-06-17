from nltk.tokenize import sent_tokenize

def get_sentences_in_generated_summary(summary, model_type):
    assert model_type in ["multi_news", "cnn_dailymail"], "Incorrect model type given."

    if model_type == "cnn_dailymail":
        summary_sentences = summary.split("<n>")
        # A case where an empty string will be the last sentence
        if len(summary_sentences[-1]) == 0:
            summary_sentences = summary_sentences[:-1]

        last_sentence_is_complete = summary_sentences[-1][-1] == "."
        
        if (not last_sentence_is_complete) and (len(summary_sentences) > 1):
            summary_sentences = summary_sentences[:-1]
        
        for i in range(len(summary_sentences)):
            if summary_sentences[i][-1] not in [".", "!", "?"]:
                summary_sentences[i] += "."
            # Some sentences tends to have space before "."
            summary_sentences[i] = summary_sentences[i].replace(" .", ".")

        return summary_sentences
    
    elif model_type == "multi_news":
        summary = summary[2:]  # Remove " -", which is always in the beginning of multinews summaries
        last_sentence_is_complete = summary[-1] == "."
        summary_sentences = sent_tokenize(summary)

        if (not last_sentence_is_complete) and (len(summary_sentences) > 1):
            summary_sentences = summary_sentences[:-1]

        for i in range(len(summary_sentences)):
            if summary_sentences[i][-1] not in [".", "!", "?"]:
                summary_sentences[i] += "."

        return summary_sentences


def make_sentence_chunks(all_sentences):
    chunks = []
    current_chunk = []
    num_tokens_in_current_chunk = 0
    
    for sentence in all_sentences:
        num_tokens = len(sentence.split(" "))
        if num_tokens + num_tokens_in_current_chunk < 512:
            current_chunk.append(sentence)
            num_tokens_in_current_chunk += num_tokens
        else:
            chunks.append(current_chunk)
            current_chunk = [sentence]
            num_tokens_in_current_chunk = num_tokens
    
    # Add last chunk if there are sentences in it
    if len(current_chunk) != 0:
        chunks.append(current_chunk)
    
    return chunks


def summarize_chunk(chunk, summarizer, min_summary_length, max_summary_length):
    combined_text = " ".join(chunk)
    summary_objects = summarizer(
                            combined_text,
                            min_length = min_summary_length,
                            max_length = max_summary_length
                        )
    summarized_text = summary_objects[0]["summary_text"]
    return summarized_text


def combine_summary_sentences(sentences):
    # Some sentences dont end with punctuation
    final_sentences = []
    for sentence in sentences:
        if sentence[-1] in [".", "!", "?"]:
            final_sentences.append(sentence)
        else:
            final_sentences.append(sentence + ".")

    return " ".join(final_sentences)
