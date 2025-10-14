import torch
import pandas as pd
import json
from .preprocess import DataProcessor
import argparse
from .device import device


class InferenceModel:
    def __init__(self, model, vocabulary):
        self.model = model
        self.preprocessor = DataProcessor(
            [], mask=False, vocabulary=vocabulary
        )

    def _infer(self, x: torch.tensor) -> torch.tensor:
        self.model.eval()
        with torch.no_grad():
            y = torch.exp(self.model(x)[-1, :])
        return y

    def get_inference_results(self, sentence: str):
        seq = self.preprocessor.normalize_line(sentence)
        seq = self.preprocessor.translate(seq)
        x = torch.tensor(seq)
        y = self._infer(x)
        results = pd.DataFrame(
            index=self.preprocessor.vocabulary,
            data=y,
            columns=["probability"],
        ).sort_values(by="probability", ascending=False)
        return results

    def infer(self, sentence: str, top_k=3):
        if sentence[-1] == " ":
            intermediate_results = self.get_inference_results(sentence)
            intermediate_results = intermediate_results.loc[
                intermediate_results.index != "NULL"
            ]
            return (
                intermediate_results.head(top_k)
                / intermediate_results["probability"].sum()
            )
        else:
            *sentence, first_letters = sentence.split()
            sentence = " ".join(sentence)
            intermediate_results = self.get_inference_results(sentence)
            intermediate_results = intermediate_results.loc[
                intermediate_results.index.str.startswith(first_letters)
            ]
            intermediate_results = intermediate_results.loc[
                intermediate_results.index != "NULL"
            ]
            return (
                intermediate_results.head(top_k)
                / intermediate_results["probability"].sum()
            )


def infer_entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", type=str, help="Input sentence")
    parser.add_argument(
        "--model", type=str, help="Path to model", default="model.pt"
    )
    parser.add_argument(
        "--vocab",
        type=str,
        help="Path to vocabulary",
        default="vocabulary.json",
    )
    args = parser.parse_args()

    try:
        vocab = json.load(open(args.vocab, "r"))
    except FileNotFoundError:
        print(
            f"\033[31mError\033[0m: Vocabulary file {args.vocab} not found."
        )
        return
    try:
        model = torch.load(
            args.model,
            weights_only=False,
            map_location=torch.device(device),
        )
    except FileNotFoundError:
        print(
            f"\033[31mError\033[0m: Model file {args.model} not found."
        )
        return

    preprocessor = DataProcessor([], mask=False, vocabulary=vocab)

    inference_model = InferenceModel(model, vocab)
    results = inference_model.infer(args.sentence, top_k=3)

    print(results)
