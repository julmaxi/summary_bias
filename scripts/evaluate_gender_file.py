import argparse
from pathlib import Path

from summarybias.evaluate.wordlists import load_helm_gender_word_list_counter
from summarybias.evaluate.name_classifier import ListBasedNameGenderClassifier

from summarybias.evaluate.load import Instance

from summarybias.evaluate.measures import compute_word_list_score_with_ci, compute_entity_inclusion_score_with_ci, compute_entity_hallucination_score_with_ci

from summarybias.evaluate.alignment import AlignedInstance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    args = parser.parse_args()

    word_list_counter = load_helm_gender_word_list_counter()
    gender_classifier = ListBasedNameGenderClassifier.from_census_data()

    instances = Instance.from_jsonl(args.input_file)

    word_list_bias, word_list_ci = compute_word_list_score_with_ci(instances, word_list_counter)
    print(f"Word list bias: {word_list_bias:.3f} ({word_list_ci.format_for_cli()})")

    aligned_instances = [
        AlignedInstance.from_instance(instance) for instance in instances
    ]

    entity_inclusion_bias, entity_inclusion_ci, (entity_inclusion_max_key, entity_inclusion_min_key) = compute_entity_inclusion_score_with_ci(
        ["male", "female"],
        aligned_instances
    )
    print(f"Inclusion bias: {entity_inclusion_bias:.3f} ({entity_inclusion_ci.format_for_cli()})")
    print("Inclusion bias direction", entity_inclusion_max_key, entity_inclusion_min_key)

    entity_hallucination_bias, entity_hallucination_ci, most_frequent_cat = compute_entity_hallucination_score_with_ci(
        aligned_instances,
        gender_classifier
    )

    print(f"Hallucination bias: {entity_hallucination_bias:.3f} ({entity_hallucination_ci.format_for_cli()})")
    print("Most frequent hallucination category", most_frequent_cat)


if __name__ == "__main__":
    main()