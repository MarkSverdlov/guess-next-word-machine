import pytest


def test_normalize_corpus():
    from guesser.preprocess import normalize_corpus

    input_lines = [
        "Hello, World!",
        "This is a test.",
        "Normalization's fun!!!",
        "12345 and symbols #@$%^&*()",
        "Mixed CASE and ``quotes''",
    ]
    expected_output = [
        ["hello", "world"],
        ["this", "is", "a", "test"],
        ["normalization's", "fun"],
        ["12345", "and", "symbols"],
        ["mixed", "case", "and", "quotes"],
    ]
    assert normalize_corpus(input_lines) == expected_output
