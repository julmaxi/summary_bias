import argparse
from pathlib import Path

from summarybias.evaluate.load import Instance
from summarybias.evaluate.alignment import AlignedInstance
from collections import Counter
import pprint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    args = parser.parse_args()

    instances = Instance.from_jsonl(args.input_file)

    aligned_instances = [
        AlignedInstance.from_instance(instance) for instance in instances
    ]

    last_name_counter = Counter()
    first_name_counter = Counter()
    first_name_inclusion_counter = Counter()
    last_name_inclusion_counter = Counter()

    for instance in aligned_instances:
        for id_, instruction in instance.instructions.items():
            if instruction.entity_category is not None:
                last_name_counter.update([instruction.last_name])
                first_name_counter.update([instruction.first_name])

            if len(instance.source_to_summary_alignment.get(id_, [])) > 0:
                first_name_inclusion_counter.update([instruction.first_name])
                last_name_inclusion_counter.update([instruction.last_name])

    pprint.pprint(last_name_counter.most_common(10))
    pprint.pprint(first_name_counter.most_common(10))

    first_name_inclusion_freqs = []
    for name, include_cnt in first_name_inclusion_counter.items():
        first_name_inclusion_freqs.append(
            (include_cnt / first_name_counter[name], name)
        )
    first_name_inclusion_freqs.sort(reverse=True)
    pprint.pprint(first_name_inclusion_freqs[:10])

    last_name_inclusion_freqs = []
    for name, include_cnt in last_name_inclusion_counter.items():
        last_name_inclusion_freqs.append(
            (include_cnt / last_name_counter[name], name)
        )
    last_name_inclusion_freqs.sort(reverse=True)
    pprint.pprint(last_name_inclusion_freqs[:10])
