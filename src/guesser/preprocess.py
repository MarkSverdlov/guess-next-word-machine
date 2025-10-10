import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter
import argparse
import torch
from .device import device


class DataProcessor:
    def __init__(self, lines: list[str], mask=True, vocabulary=None):
        self.lines = lines
        self.normalize_corpus()
        if vocabulary is not None:
            self.vocabulary = vocabulary
        else:
            self.vocabulary = self.extract_vocabulary()
        self.translation_table = {
            word: idx for idx, word in enumerate(self.vocabulary)
        }
        if mask:
            self.mask_data()

    def translate(self, seq):
        rval = []
        for word in seq:
            if word in self.translation_table:
                rval.append(self.translation_table[word])
            else:
                rval.append(self.translation_table["NULL"])
        return rval

    def normalize_corpus(self):
        allowed_characters = set(
            "abcdefghijklmnopqrstuvwxyz0123456789'"
        )

        self.lines = [line.lower() for line in self.lines]
        self.lines = [
            "".join(
                c
                for c in line
                if c in allowed_characters or c.isspace()
            )
            for line in self.lines
        ]
        self.lines = [line.replace("''", "") for line in self.lines]
        self.lines = [line.split() for line in self.lines]

    @staticmethod
    def mask_word(word, alpha=1e-3):
        if random.random() < alpha:
            return "NULL"
        return word

    def mask_data(self, alpha=1e-3):
        self.lines = [
            [self.mask_word(word, alpha) for word in line]
            for line in self.lines
        ]

    def extract_vocabulary(self):
        vocabulary = set()
        for line in self.lines:
            vocabulary.update(line)
        vocabulary = ["NULL"] + list(sorted(vocabulary))
        return vocabulary

    def get_random_sequence(self, line_length):
        if all(len(line) < line_length for line in self.lines):
            raise ValueError(
                "All lines are shorter than the specified line_length."
            )
        candidates = [
            line for line in self.lines if len(line) >= line_length
        ]
        line = random.choice(candidates)
        start_index = random.randint(0, len(line) - line_length)
        sequence = line[start_index : start_index + line_length]
        return sequence

    def get_data_batch(self, line_length, batch_size):
        Xs = []
        ys = []
        for _ in range(batch_size):
            seq = self.get_random_sequence(line_length)
            translated_seq = self.translate(seq)
            Xs.append(torch.tensor(translated_seq[:-1], device=device))
            ys.append(torch.tensor(translated_seq[1:], device=device))
        Xs = torch.stack(Xs)
        ys = torch.stack(ys)
        return Xs, ys

    # def data_record_generator(self):
    #         while True:
    #             line = random.choice(self.lines)
    #             if len(line) < 2:
    #                 continue
    #             start_index = random.randint(0, len(line) - 1)
    #             seq_length = random.randint(1, len(line) - start_index)
    #             input_seq = line[start_index : start_index + seq_length - 1]
    #             output_seq = line[start_index + 1: start_index + seq_length]
    #             yield input_seq, output_seq
    # def generate_data(self, num_records=200):
    #     data = [
    #         (input_seq, output_seq)
    #         for (input_seq, output_seq), _ in zip(
    #             self.data_record_generator(), range(num_records)
    #         )
    #     ]
    #     return data


def chart_statistics(lines):
    lengths = np.array([len(line) for line in lines])
    mean = lengths.mean()
    median = np.median(lengths)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(lengths, bins=50)
    ax.text(
        0.3,
        0.8,
        f"Mean: {mean:.2f}",
        color="red",
        transform=plt.gca().transAxes,
        size=12,
        weight="bold",
    )
    ax.text(
        0.3,
        0.7,
        f"Median: {median:.2f}",
        color="blue",
        transform=plt.gca().transAxes,
        size=12,
        weight="bold",
    )
    fig.savefig("length_distribution.png")

    word_counter = Counter()
    for line in lines:
        word_counter.update(line)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()
    number_of_unique_words = len(word_counter.keys())
    fig.suptitle(
        f"Histogram of Word Frequencies (Number of Unique Words: {number_of_unique_words})"
    )
    most_common_words = dict(word_counter.most_common(20))
    common_words = dict(word_counter.most_common(40)[20:])
    axes[0].set_title("20 Most Common Words")
    axes[0].bar(most_common_words.keys(), most_common_words.values())
    axes[0].xaxis.set_tick_params(rotation=45)
    axes[1].set_title("21-40 Most Common Words")
    axes[1].bar(common_words.keys(), common_words.values())
    axes[1].xaxis.set_tick_params(rotation=45)
    most_common_words = dict(word_counter.most_common(60)[40:])
    common_words = dict(word_counter.most_common(80)[60:])
    axes[2].set_title("41-60 Most Common Words")
    axes[2].bar(most_common_words.keys(), most_common_words.values())
    axes[2].xaxis.set_tick_params(rotation=45)
    axes[3].set_title("61-80 Most Common Words")
    axes[3].bar(common_words.keys(), common_words.values())
    axes[3].xaxis.set_tick_params(rotation=45)
    fig.savefig("word_frequency_distribution.png")


def chart_statistics_entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    args = parser.parse_args()
    with open(args.corpus, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_preprocessor = DataProcessor(lines, mask=False)
    chart_statistics(data_preprocessor.lines)
