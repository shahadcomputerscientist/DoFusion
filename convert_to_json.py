import csv
import pandas as pd
import json

df = pd.read_csv("data/findit2/train.txt")

with open("data/findit2/train.jsonl", "w") as f:
    for _, row in df.iterrows():

        record = {
            "id": row["image"].replace(".png", ""),
            "is_forged": int(row["forged"])
        }

        f.write(json.dumps(record) + "\n")

print("train.jsonl created!")


input_file = "data/findit2/test.txt"
output_file = "data/findit2/test.jsonl"

with open(input_file, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)

    with open(output_file, "w", encoding="utf-8") as out:
        for row in reader:
            image_name = row["image"].replace(".png", "")
            json_line = {"id": image_name}
            out.write(json.dumps(json_line) + "\n")

print("test.jsonl created successfully")