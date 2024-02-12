import argparse
from pathlib import Path
from dataclasses import dataclass

from summarybias.evaluate.wordlists import load_helm_gender_word_list_counter
from summarybias.evaluate.name_classifier import ListBasedNameGenderClassifier

from summarybias.evaluate.load import Instance

from summarybias.evaluate.measures import compute_word_list_score_with_ci, compute_entity_inclusion_score_with_ci, compute_entity_hallucination_score_with_ci, ConfidenceInterval, WordListCounter, ListBasedNameGenderClassifier

from summarybias.evaluate.alignment import AlignedInstance

from typing import Union

from summarybias.evaluate.distinguishability import compute_distinguishability_with_ci, BoWEmbedder, TransformerEmbedder

BALANCED_FILE_PATTERN = "legacy_summaries/onto-nw_gender-balanced_10.jsonl_{model}_summaries.jsonl"
MONO_FILE_PATTERN = "legacy_summaries/onto-nw_gender_mono_10.jsonl_{model}_summaries.jsonl"

SUMMARIZERS = [
    ("facebook-bart-large-cnn", "BART CNN/DM"),
    ("facebook-bart-large-xsum", "BART XSum"),
    ("google-pegasus-cnn_dailymail", "Pegasus CNN/DM"),
    ("google-pegasus-xsum", "Pegasus XSum"),
    ("meta-llama-Llama-2-7b-chat-hf", "Llama 7b"),
    ("meta-llama-Llama-2-13b-chat-hf", "Llama 13b"),
]

class BoundedResultCell:
    def __init__(self, value: float, bounds: ConfidenceInterval) -> None:
        self.value = value
        self.bounds = bounds
    
    def format_for_cli(self) -> str:
        val = f"{self.value:.3f}"
        if self.bounds is not None:
            bounds = {self.bounds.format_for_cli()}
            return f"{val} ({bounds})"
        else:
            return val


class TextCell:
    def __init__(self, text: str) -> None:
        self.text = text
    
    def format_for_cli(self) -> str:
        return self.text

class ResultsTable:
    def __init__(self, headers: list[str], rows: list[list[Union[TextCell, BoundedResultCell]]]) -> None:
        self.rows = rows
        self.headers = headers
    
    def format_for_cli(self) -> str:
        all_rows = [self.headers] + [[cell.format_for_cli() for cell in row] for row in self.rows]

        max_col_widths = [max(len(row[i]) for row in all_rows) for i in range(len(self.headers))]

        all_rows = [
            "\t".join(
                cell.ljust(max_col_widths[i]) for i, cell in enumerate(row)
            ) for row in all_rows
        ]
        
        return "\n".join(all_rows)


def main():
    rows = []

    headers = ["Measure"] + [label for _, label in SUMMARIZERS]

    bootstrap_samples = None

    word_list_counter = load_helm_gender_word_list_counter()
    name_classifier = ListBasedNameGenderClassifier.from_census_data()

    bert_embedder = TransformerEmbedder()
    bow_embedder = BoWEmbedder()

    table = [
        [TextCell("Word List Inclusion")],
        [TextCell("Entity Inclusion")],
        [TextCell("Entity Hallucination")],
        [TextCell("Distinguishability (Count)")],
        [TextCell("Distinguishability (Dense)")]
    ]

    for model, _ in SUMMARIZERS:
        input_file = Path(BALANCED_FILE_PATTERN.format(model=model))
        instances = Instance.from_jsonl(input_file)
        aligned_instances = [
            AlignedInstance.from_instance(instance) for instance in instances
        ]
        hallucination_bias, hallucination_ci = compute_entity_hallucination_score_with_ci(
            aligned_instances,
            name_classifier,
            n_bootrap_samples=bootstrap_samples
        )

        word_list_bias, word_list_bias_ci = compute_word_list_score_with_ci(
            instances,
            word_list_counter,
            n_bootrap_samples=bootstrap_samples)

        entity_inclusion_bias, entity_inclusion_ci = compute_entity_inclusion_score_with_ci(
            ["male", "female"],
            aligned_instances,
            n_bootrap_samples=bootstrap_samples
        )

        table[0].append(BoundedResultCell(word_list_bias, word_list_bias_ci))
        table[1].append(BoundedResultCell(entity_inclusion_bias, entity_inclusion_ci))
        table[2].append(BoundedResultCell(hallucination_bias, hallucination_ci))

        input_file = Path(MONO_FILE_PATTERN.format(model=model))

        instances = Instance.from_jsonl(input_file)

        bow_distinguishability = compute_distinguishability_with_ci(
            instances,
            bow_embedder,
            n_bootrap_samples=bootstrap_samples
        )

        transformers_distinguishability = compute_distinguishability_with_ci(
            instances,
            bert_embedder,
            n_bootrap_samples=bootstrap_samples
        )

        table[3].append(BoundedResultCell(bow_distinguishability[0], bow_distinguishability[1]))
        table[4].append(BoundedResultCell(transformers_distinguishability[0], transformers_distinguishability[1]))
    
    print(ResultsTable(headers, table).format_for_cli())

if __name__ == "__main__":
    main()