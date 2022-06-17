# %%
from pathlib import Path
from pandas import read_csv

model_type = "multi_news"
# model_type = "cnn_dailymail"

filenames = {
    "cnn_dailymail": "eval_cnn.txt",
    "multi_news": "eval_mn.txt"
}

df = read_csv(Path(__file__).parent / filenames[model_type], delimiter=";")

# df = df.tail(1000)
# df = df.head(1000)

print(model_type.upper())
print("Average rouge1 precision: ", round(df["rouge1_precision"].mean(), 4))
print("Average rouge1 recall:    ", round(df["rouge1_recall"].mean(), 4))
print("Average rouge1 f1:        ", round(df["rouge1_f1"].mean(), 4))
print("Average rouge1 f1 std:    ", round(df["rouge1_f1"].std(), 4))
print("")
print("Average rouge2 precision: ", round(df["rouge2_precision"].mean(), 4))
print("Average rouge2 recall:    ", round(df["rouge2_recall"].mean(), 4))
print("Average rouge2 f1:        ", round(df["rouge2_f1"].mean(), 4))
print("Average rouge2 f1 std:    ", round(df["rouge2_f1"].std(), 4))
print("")
print("Average rougeL precision: ", round(df["rougeL_precision"].mean(), 4))
print("Average rougeL recall:    ", round(df["rougeL_recall"].mean(), 4))
print("Average rougeL f1:        ", round(df["rougeL_f1"].mean(), 4))
print("Average rougeL f1 std:    ", round(df["rougeL_f1"].std(), 4))

# %%
