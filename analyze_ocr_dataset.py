import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter


def load_data(source_file):
    with open(source_file, "r") as file:
        data = json.load(file)
    return data["ocr_files"]


def split_data(data):
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, val_data


def analyze_data(data):
    text_lengths = [len(item["text"]) for item in data]
    unique_texts = set(item["text"] for item in data)
    length_counts = Counter(text_lengths)
    return length_counts, unique_texts


def visualize_data(train_lengths, val_lengths, train_unique_texts, val_unique_texts):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.hist(
        list(train_lengths.keys()),
        bins=20,
        alpha=0.7,
        label="Train",
        weights=list(train_lengths.values()),
    )
    plt.hist(
        list(val_lengths.keys()),
        bins=20,
        alpha=0.7,
        label="Validation",
        weights=list(val_lengths.values()),
    )
    plt.xlabel("Length of Text")
    plt.ylabel("Frequency")
    plt.title("Text Length Distribution")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(["Train", "Validation"], [len(train_unique_texts), len(val_unique_texts)])
    plt.xlabel("Dataset")
    plt.ylabel("Number of Unique Texts")
    plt.title("Unique Texts in Datasets")

    plt.tight_layout()
    plt.show()


def output_analysis(length_counts, unique_texts, total_samples, dataset_name):
    print(f"{dataset_name} Dataset Analysis")
    print("=" * len(f"{dataset_name} Dataset Analysis"))
    print(f"Total samples: {total_samples}")
    print("Text Length Distribution:")
    for length, count in length_counts.items():
        print(
            f"Length {length}: {count} samples ({(count / total_samples) * 100:.2f}%)"
        )
    print(
        f"Number of unique texts: {len(unique_texts)} ({(len(unique_texts) / total_samples) * 100:.2f}%)"
    )
    print("\n")


def main(source_file):
    data = load_data(source_file)
    train_data, val_data = split_data(data)

    train_length_counts, train_unique_texts = analyze_data(train_data)
    val_length_counts, val_unique_texts = analyze_data(val_data)

    visualize_data(
        train_length_counts, val_length_counts, train_unique_texts, val_unique_texts
    )
    output_analysis(train_length_counts, train_unique_texts, len(train_data), "Train")
    output_analysis(val_length_counts, val_unique_texts, len(val_data), "Validation")


if __name__ == "__main__":
    source_file = "/content/ocr.json"
    main(source_file)
