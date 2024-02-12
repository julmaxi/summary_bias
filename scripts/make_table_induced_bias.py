import argparse
from pathlib import Path
from dataclasses import dataclass

from summarybias.evaluate.wordlists import load_helm_gender_word_list_counter
from summarybias.evaluate.name_classifier import ListBasedNameGenderClassifier

from summarybias.evaluate.load import Instance

from summarybias.evaluate.measures import compute_word_list_score_with_ci, compute_entity_inclusion_score_with_ci, compute_entity_hallucination_score_with_ci, ConfidenceInterval

from summarybias.evaluate.alignment import AlignedInstance

from typing import Union

from summarybias.evaluate.distinguishability import compute_distinguishability_with_ci, BoWEmbedder, TransformerEmbedder

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
    
    def format_for_latex(self) -> str:
        if self.bounds is not None:
            bounds = self.bounds.format_for_latex()
            return f"\\renewcommand{{\\arraystretch}}{{0.6}}\\begin{{tabular}}{{@{{}}c@{{}}}} {self.value:.2f} {bounds} \end{{tabular}}"
        else:
            return f"{self.value:.2f}"


class TextCell:
    def __init__(self, text: str) -> None:
        self.text = text
    
    def format_for_cli(self) -> str:
        return self.text
    
    def format_for_latex(self) -> str:
        return self.text.replace("_", "\\_")

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
    
    def format_for_latex(self) -> str:
        all_rows = [self.headers] + [[cell.format_for_latex() for cell in row] for row in self.rows]

        all_rows = [
            " & ".join(row) for row in all_rows
        ]
        
        return "\\\\\n".join(all_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--latex", action="store_true", default=False)
    parser.add_argument("-b", "--bootstrap", action="store_true", default=False)
    args = parser.parse_args()
    rows = []

    headers = ["Measure", "\\balanced", "\\mono"]

    bootstrap_samples = 1000 if args.bootstrap else None

    balanced_file = "parsed_summaries/onto-nw_gender-balanced_10.jsonl_meta-llama-Llama-2-13b-chat-hf_bias_female_summaries.jsonl"
    mono_file = "parsed_summaries/onto-nw_gender_mono_10.jsonl_meta-llama-Llama-2-13b-chat-hf_bias_female_summaries.jsonl"

    results = {}

    word_list = load_helm_gender_word_list_counter()

    for file, label in [(balanced_file, "\\balanced"), (mono_file, "\\mono")]:
        instances = Instance.from_jsonl(file)
        aligned_instances = [
            AlignedInstance.from_instance(instance) for instance in instances
        ]
        entity_inclusion_bias, entity_inclusion_ci, (max_key, min_key) = compute_entity_inclusion_score_with_ci(
            ["male", "female"],
            aligned_instances,
            n_bootrap_samples=bootstrap_samples
        )

        word_list_bias, word_list_bias_ci = compute_word_list_score_with_ci(
            instances,
            word_list,
            n_bootrap_samples=bootstrap_samples
        )
        results[label] = {
            "word_list": (word_list_bias, word_list_bias_ci),
            "entity_inclusion": (entity_inclusion_bias, entity_inclusion_ci)
        }

    
    rows = [
        [
            TextCell("Word List"),
            BoundedResultCell(*results["\\balanced"]["word_list"]),
            BoundedResultCell(*results["\\mono"]["word_list"])
        ],
        [
            TextCell("Entity Incl."),
            BoundedResultCell(*results["\\balanced"]["entity_inclusion"]),
            BoundedResultCell(*results["\\mono"]["entity_inclusion"])
        ]
    ]


    if args.latex:
        print(ResultsTable(headers, rows).format_for_latex())
    else:
        print(ResultsTable(headers, rows).format_for_cli())

if __name__ == "__main__":
    main()