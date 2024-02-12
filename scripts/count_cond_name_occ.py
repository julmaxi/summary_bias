
from summarybias.instancegen.names import BBQNameGeneratorFactory
from summarybias.evaluate.load import Instance
from summarybias.evaluate.alignment import AlignedInstance
from collections import Counter, defaultdict

import argparse

from nltk import word_tokenize
from summarybias.evaluate.distinguishability import anonymize_summary

def flatten(l):
    return [item for sublist in l for item in sublist]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    fact = BBQNameGeneratorFactory()
    names = flatten(fact.first_names["black"].values())\
        + flatten(fact.first_names["white"].values())\
        + fact.last_names["black"]\
        + fact.last_names["white"]

    names = [name.lower() for name in names]

    name_counts = defaultdict(Counter)

    orig_instances = Instance.from_jsonl(args.path)
    instances = [AlignedInstance.from_instance(inst) for inst in orig_instances]

    last_name_counter = Counter()

    for idx, inst in enumerate(instances):
        #tokens = word_tokenize(inst.summary.lower())
        #entities = [i for i in inst.parse.named_entities if i.label == "PERSON"]
        aligned_entities = [idx for idx, al in inst.source_to_summary_alignment.items() if len(al) > 0]

        anon_sum = anonymize_summary(orig_instances[idx])
        tokens = word_tokenize(anon_sum)
        if "LAST_NAME" in tokens:
            last_name_counter.update([inst.text_category])

        c = Counter()
        for entity in aligned_entities:
            e = inst.instructions[entity]
            c.update([e.first_name.lower()] if e.first_name else [])
            c.update([e.last_name.lower()] if e.last_name else [])

            #if e.last_name == "Coleman":
                #print(orig_instances[idx].summary)
                #print(orig_instances[idx].source)
            #    print()
        
        for name in names:
            name_counts[name][inst.text_category] += c[name]
    
    print(last_name_counter)
    
    per_cat_count = Counter()

    for v in name_counts.values():
        per_cat_count.update(v)

    print(per_cat_count)
    for name, counts in name_counts.items():
        print(name, counts)
        
            
