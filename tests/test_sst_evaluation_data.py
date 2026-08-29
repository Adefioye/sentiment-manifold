from datasets import Dataset, DatasetDict
import pytest

from sentiment_manifold.data.sst import load_processed_sst_candidates


def test_load_processed_sst_candidates_requires_pythia_correct_classification_prompts(tmp_path):
    config_path = tmp_path / "tigges_pythia_correct"
    DatasetDict(
        {
            "test": Dataset.from_list(
                [
                    {
                        "example_id": "sst-1",
                        "text": "A wonderful film.",
                        "prompt": "Review Text: A wonderful film. Review Sentiment:",
                        "label": 1,
                        "sentiment_score": 0.9,
                        "split": "test",
                        "binarization_method": "tigges",
                        "pythia_correct": True,
                        "pythia_raw_num_tokens": 5,
                        "pythia_prompt_num_tokens": 10,
                    }
                ]
            )
        }
    ).save_to_disk(config_path)
    examples = load_processed_sst_candidates(tmp_path)
    assert len(examples) == 1
    assert examples[0].text == "Review Text: A wonderful film. Review Sentiment:"
    assert examples[0].label == 1
    assert examples[0].metadata["pythia_correct"] is True


def test_load_processed_sst_candidates_rejects_unfiltered_rows(tmp_path):
    config_path = tmp_path / "tigges_pythia_correct"
    DatasetDict(
        {
            "test": Dataset.from_list(
                [
                    {
                        "example_id": "sst-2",
                        "text": "A bad film.",
                        "prompt": "Review Text: A bad film. Review Sentiment:",
                        "label": 0,
                        "sentiment_score": 0.1,
                        "split": "test",
                        "pythia_correct": False,
                    }
                ]
            )
        }
    ).save_to_disk(config_path)
    with pytest.raises(ValueError, match="Pythia-correct"):
        load_processed_sst_candidates(tmp_path)
