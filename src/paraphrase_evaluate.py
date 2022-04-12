import csv
import sys
import random

RESULT_PATH = "model_results.csv"

with open(RESULT_PATH, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    keys = next(reader)
    sentences = []
    for line in reader:
        sentences.append([(keys[i], line[i]) for i in range(len(line))])

scores = {
    "bert": 0,
    "hop1": 0,
    "hop3": 0,
    "hop5": 0
}

try:
    random.shuffle(sentences)
    for sentence in sentences:
        original = sentence[0][1]
        sentence = sentence[1:]
        random.shuffle(sentence)
        print(original)
        print("\nWhich of the following is the most different from the sentence above while keeping the closest meaning?\n")
        for i in range(len(sentence)):
            print(f"{i+1}: {sentence[i][1].lower().replace('i̇', 'i')}")
        choice = int(input("Index: "))
        scores[sentence[choice-1][0]] += 1
        print("\n")
except Exception as e:
    pass

print(scores)