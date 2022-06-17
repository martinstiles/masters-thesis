# %%
from pathlib import Path
from pandas import read_csv

df = read_csv(Path(__file__).parent / "eval2.txt", delimiter=";")

# df = df.head(3000)

print("Average rouge1 precision: ", round(df["rouge1_precision"].mean(),4))
print("Average rouge1 recall:    ", round(df["rouge1_recall"].mean(),4))
print("Average rouge1 f1:        ", round(df["rouge1_f1"].mean(),4))
print("Average rouge1 f1 std:    ", round(df["rouge1_f1"].std(),4))
print("")
print("Average rouge2 precision: ", round(df["rouge2_precision"].mean(),4))
print("Average rouge2 recall:    ", round(df["rouge2_recall"].mean(),4))
print("Average rouge2 f1:        ", round(df["rouge2_f1"].mean(),4))
print("Average rouge2 f1 std:    ", round(df["rouge2_f1"].std(),4))
print("")
print("Average rougeL precision: ", round(df["rougeL_precision"].mean(),4))
print("Average rougeL recall:    ", round(df["rougeL_recall"].mean(),4))
print("Average rougeL f1:        ", round(df["rougeL_f1"].mean(),4))
print("Average rougeL f1 std:    ", round(df["rougeL_f1"].std(),4))

# %%
