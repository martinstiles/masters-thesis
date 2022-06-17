# %%
from pandas import read_csv
from pathlib import Path
from collections import Counter


ARTICLES_PATH = Path(__file__).parent.parent / "data" / "clusters_large.txt"
ENTITIES_PATH = Path(__file__).parent.parent / "data" / "entities_large.txt"

df_articles = read_csv(ARTICLES_PATH, sep=";")
# df_articles = df_articles.loc[df_articles["clusterId"] == cluster_id]
# df_articles = df_articles[["title", "summary", "body"]]

# Load entities
df_entities = read_csv(ENTITIES_PATH, sep=";")

# %%


def print_entity_stats(df_entities):
    num_entities_in_clusters = []
    for cluster_id in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]:
        df_current = df_entities.loc[df_entities["clusterId"] == cluster_id]
        df_current = df_current["wikidataId"]
        current_entities = set()
        for entity_id in df_current:
            current_entities.add(entity_id)
        num_entities_in_clusters.append(len(current_entities))
        
    print(f"Average number of unique entities per cluster: {round(sum(num_entities_in_clusters)/len(num_entities_in_clusters), 0)}")

print_entity_stats(df_entities)


# %%
def print_num_mentions_of_top_ten_entities(df_entities):
    entities_2D = []
    for cluster_id in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]:
        df_current = df_entities.loc[df_entities["clusterId"] == cluster_id]
        df_current = df_current["wikidataId"]
        current_entities = []
        for entity_id in df_current:
            current_entities.append(entity_id)
        entities_2D.append(current_entities)
    
    counters = [Counter(entities) for entities in entities_2D]
    top_5s = [counter.most_common(5) for counter in counters]
    avg_num_mentions_of_top_5_entities = [sum([tup[1] for tup in top_5])/len(top_5) for top_5 in top_5s]
    
    print(f"Average number of mentions for the top 5 entities in a cluster {round(sum(avg_num_mentions_of_top_5_entities)/len(avg_num_mentions_of_top_5_entities), 0)}")


print_num_mentions_of_top_ten_entities(df_entities)

# %%
def print_num_unique_articles_in_cluster(df_articles):
    num_unique_articles = []
    for cluster_id in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]:
        df_current = df_articles.loc[df_articles["clusterId"] == cluster_id]
        df_titles = df_current["title"]
        unique_titles = set()
        for title in df_titles:
            unique_titles.add(title)
        num_unique_articles.append(len(unique_titles))

    print(f"Average number of unique articles in a cluster: {(round(sum(num_unique_articles)/len(num_unique_articles),5))}")

print_num_unique_articles_in_cluster(df_articles)


# %%
def print_avg_num_tokens_in_articles(df_articles):
    num_tokens_in_clusters = []
    for cluster_id in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]:
        df_current = df_articles.loc[df_articles["clusterId"] == cluster_id]
        df_bodys = df_current["body"]
        num_tokens_in_cluster = 0
        for body in df_bodys:
            num_tokens_in_article = len(body.split(" "))
            num_tokens_in_cluster += num_tokens_in_article
        num_tokens_in_clusters.append(num_tokens_in_cluster/len(df_bodys))
    
    print(f"Average number of tokens in an article: {round(sum(num_tokens_in_clusters)/len(num_tokens_in_clusters),0)}")

print_avg_num_tokens_in_articles(df_articles)

# %%
"""
TODO
-> avg in a cluster (avg*avg num unique articles) = 458 * 21 = 9618

"""