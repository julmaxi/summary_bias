import argparse
from pathlib import Path

from summarybias.evaluate.distinguishability import BoWEmbedder, TransformerEmbedder, compute_distinguishability_with_ci
from summarybias.evaluate.load import Instance

if __name__ == "__main__":
    bert_embedder = TransformerEmbedder()
    bow_embedder = BoWEmbedder()

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    args = parser.parse_args()

    instances = Instance.from_jsonl(args.input_file)

    from collections import Counter
    original_article_ids = Counter(instance.original_article_id for instance in instances)

    instances = [instance for instance in instances if original_article_ids[instance.original_article_id] == max(original_article_ids.values())]

    bow_distinguishability, _ = compute_distinguishability_with_ci(
        instances,
        bow_embedder,
        n_bootrap_samples=None
    )

    print(f"BoW distinguishability: {bow_distinguishability:.3f}")

    bert_distinguishability, _ = compute_distinguishability_with_ci(
        instances,
        bert_embedder,
        n_bootrap_samples=None
    )

    print(f"BERT distinguishability: {bert_distinguishability:.3f}")
