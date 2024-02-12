import argparse
from pathlib import Path
from dataclasses import dataclass

from summarybias.evaluate.wordlists import load_helm_gender_word_list_counter
from summarybias.evaluate.name_classifier import ListBasedNameGenderClassifier

from summarybias.evaluate.load import Instance

from summarybias.evaluate.measures import compute_word_list_score_with_ci, compute_entity_inclusion_score_with_ci, compute_entity_hallucination_score_with_ci, ConfidenceInterval

from summarybias.evaluate.alignment import AlignedInstance

from collections import Counter
from typing import Union

from summarybias.evaluate.distinguishability import compute_distinguishability_with_ci, BoWEmbedder, TransformerEmbedder

FILE_PATTERNS = [
    ("parsed_summaries/onto-nw_race_{mode}_10.jsonl_{model}_summaries.jsonl", "Random"),
    ("parsed_summaries/onto-nw_race_black:m,white:f_{mode}_10.jsonl_{model}_summaries.jsonl", "Black Male/White Female"),
    ("parsed_summaries/onto-nw_race_black:m,white:m_{mode}_10.jsonl_{model}_summaries.jsonl", "Black Male/White Male"),
    ("parsed_summaries/onto-nw_race_black:f,white:m_{mode}_10.jsonl_{model}_summaries.jsonl", "Black Female/White Male"),
    ("parsed_summaries/onto-nw_race_black:f,white:f_{mode}_10.jsonl_{model}_summaries.jsonl", "Black Female/White Female"),
]

SUMMARIZERS = [
    ("facebook-bart-large-cnn", "BART CNN/DM"),
    ("facebook-bart-large-xsum", "BART XSum"),
    ("google-pegasus-xsum", "Pegasus XSum"),
    ("google-pegasus-cnn_dailymail", "Pegasus CNN/DM"),
    ("meta-llama-Llama-2-7b-chat-hf", "Llama2 7b"),
    ("meta-llama-Llama-2-13b-chat-hf", "Llama2 7b")
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

    headers = ["Corpus"] + [label for _, label in SUMMARIZERS]

    bootstrap_samples = 1000 if args.bootstrap else None

    bert_embedder = TransformerEmbedder()
    bow_embedder = BoWEmbedder()

    permissible_article_ids = None
    all_instances = {}
    for file_pattern, corpus_label in FILE_PATTERNS:
        for model, model_name in SUMMARIZERS:
            instances = Instance.from_jsonl(Path(file_pattern.format(mode="balanced", model=model)))
            all_instances[(file_pattern, model, "balanced")] = instances
            if permissible_article_ids is None:
                permissible_article_ids = set(i.original_article_id for i in instances)
            else:
                permissible_article_ids &= set(i.original_article_id for i in instances)

            instances = Instance.from_jsonl(Path(file_pattern.format(mode="mono", model=model)))

            id_counter = Counter(i.original_article_id for i in instances)

            permissible_article_ids &= set(k for k, v in id_counter.items() if v == 20)

            all_instances[(file_pattern, model, "mono")] = instances


    print("N:", len(permissible_article_ids))

    for file_pattern, corpus_label in FILE_PATTERNS:
        row = [
            TextCell(corpus_label)
        ]
        for model, model_name in SUMMARIZERS:
            input_file = Path(file_pattern.format(mode="balanced", model=model))
            instances = all_instances[(file_pattern, model, "balanced")]
            instances = [i for i in instances if i.original_article_id in permissible_article_ids]
            aligned_instances = [
                AlignedInstance.from_instance(instance) for instance in instances
            ]

            entity_inclusion_bias, entity_inclusion_ci, (max_key, min_key) = compute_entity_inclusion_score_with_ci(
                ["black", "white"],
                aligned_instances,
                n_bootrap_samples=bootstrap_samples
            )

            print("Highest inclusion in ", corpus_label, model_name, "-", max_key)

            row.append(BoundedResultCell(entity_inclusion_bias, entity_inclusion_ci))

        rows.append(row)
    
    bow_dist_rows = []
    transformers_dist_rows = []

    for file_pattern, corpus_label in FILE_PATTERNS:
        bow_row = [
            TextCell(corpus_label)
        ]
        trans_row = [
            TextCell(corpus_label)
        ]
        for model, _ in SUMMARIZERS:
            instances = all_instances[(file_pattern, model, "mono")]
            instances = [i for i in instances if i.original_article_id in permissible_article_ids]
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

            bow_row.append(BoundedResultCell(bow_distinguishability[0], bow_distinguishability[1]))
            trans_row.append(BoundedResultCell(transformers_distinguishability[0], transformers_distinguishability[1]))
        bow_dist_rows.append(bow_row)
        transformers_dist_rows.append(trans_row)


    if args.latex:
        print(ResultsTable(headers, rows + bow_dist_rows + transformers_dist_rows).format_for_latex())
    else:
        print(ResultsTable(headers, rows + bow_dist_rows + transformers_dist_rows).format_for_cli())

if __name__ == "__main__":
    main()