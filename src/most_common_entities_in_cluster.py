from pathlib import Path
# %%
from pandas import read_csv
from collections import Counter
from pathlib import Path

ENTITIES_PATH = Path(__file__).parent.parent / "data" / "entities_large.txt"

df_entities = read_csv(ENTITIES_PATH, sep=";")

# %%
for clusterId in range(15):
    df = df_entities.loc[df_entities["clusterId"] == clusterId]
    # Extract text from
    df = df["entityName"]
    counter = Counter(df)
    print("ClusterID =", clusterId)
    print(counter.most_common(10))
    print("")

# %%
