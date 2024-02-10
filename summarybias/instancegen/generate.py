from pathlib import Path
from .templates import ReplaceTemplate, ReplaceableEntity, iter_onto_files, ReplaceInstruction, MentionCategory
from .ontonotes import OntoNoteFile
from .names import BBQNameGeneratorFactory, NameGender
import re
import json

import argparse
import numpy as np
import random
import math

import itertools as it
import math
from dataclasses import asdict, dataclass

def format_text(text: str) -> str:
    text = text.replace("-LRB-", '(')
    text = text.replace("-RRB-", ')')
    text = text.replace("-AMP-", '&')
    text = text.replace("-LSB-", '[')
    text = text.replace("-RSB-", ']')
    text = text.replace("-LCB-", '{')
    text = text.replace("-RCB-", '}')
    text = text.replace("-LAB-", '<')
    text = text.replace("-RAB-", '>')
    punct_regex = re.compile(r" ([.,!?;:])")
    text = punct_regex.sub(r"\1", text)
    return text


def parse_name_config(s: str):
    out = {
        "male": "common",
        "female": "common",
        "unknown": "common"
    }

    for part in s.split(","):
        k, v = part.split(":")
        out[k] = tuple(sorted(v.split(".")))
    return out

def parse_gender_config(s: str):
    out = {}

    for part in s.split(","):
        cat, genders = part.split(":")
        out_genders = []
        for gender in genders:
            if gender == "f":
                out_gender = NameGender.FEMALE
            elif gender == "m":
                out_gender = NameGender.MALE
            else:
                raise ValueError("Invalid gender value")
            out_genders.append(out_gender)
        
        out[cat] = tuple(sorted(out_genders))
    
    return out


def is_gendered_entity(e):
    return e.first_name is not None or any(m.category in (MentionCategory.MR_MRS_MS, MentionCategory.PRONOUN_3P_INDIR, MentionCategory.PRONOUN_3P_NOM, MentionCategory.PRONOUN_3P_POSS, MentionCategory.PRONOUN_3P_REFL) for m in e.mentions)


def generate_paired_name_assignments(
        template: ReplaceTemplate,
        name_generator_factory: BBQNameGeneratorFactory,
        categories: list[str],
        replace_last_names: bool,
        name_sources: dict[str, tuple[str]],
        category_genders: dict[str, tuple[NameGender]],
    ) -> list[dict[str, ReplaceInstruction]]:
    name_sources_list = list(name_sources.values())
    modifies_name = any(l != name_sources_list[0] for l in name_sources_list[1:])
    gender_list = list(category_genders.values())
    modifies_gender = any(l != gender_list[0] for l in gender_list[1:])

    replaceable_entities = [
        e for e in template.replaceable_entities if (modifies_gender and is_gendered_entity(e)) or (modifies_name and e.first_name is not None) or (modifies_name and replace_last_names and e.last_name is not None)
    ]
    non_replaceable_entities = [
        e for e in template.replaceable_entities if e not in replaceable_entities
    ]

    entity_count = len(replaceable_entities)

    category_size = int(math.ceil(entity_count / len(categories)))

    names = name_generator_factory()

    category_name_and_gender_lists = {}

    for category in categories:
        cat_category_genders = category_genders[category]
        planned_entity_gender_assignment = [
            random.choice(cat_category_genders) for _ in range(category_size)
        ]
        cat_name_sources = name_sources[category]
        planned_entity_source_assignment = [
            random.choice(cat_name_sources) for _ in range(category_size)
        ]

        name_assignments = []

        for idx, (gender, src) in enumerate(zip(planned_entity_gender_assignment, planned_entity_source_assignment)):
            name_assignments.append((
                gender,
                names.next_first_name(
                    gender,
                    src
                ),
                names.next_last_name(src) if replace_last_names else None
            ))

        category_name_and_gender_lists[category] = name_assignments

    base_category_assignment = [idx for idx in range(len(categories)) for _ in range(category_size)]
    random.shuffle(base_category_assignment)

    if replace_last_names:
        non_replaceable_last_names = [names.next_last_name(random.choice(name_sources_list[0])) for _ in range(len(non_replaceable_entities))]
    else:
        non_replaceable_last_names = []
    for cat_assignment in it.permutations(categories, len(categories)):
        instructions = {}

        non_replaceable_last_names_iter = iter(non_replaceable_last_names)

        #print(entity_count, category_name_and_gender_lists)
        per_category_iterators = {
            k: iter(v) for k, v in category_name_and_gender_lists.items()
        }
        
        for entity, base_cat in zip(replaceable_entities, base_category_assignment):
            entity_cat = cat_assignment[base_cat]
            entity_gender = None
            first_name = None
            entity_gender, first_name, last_name = next(per_category_iterators[entity_cat])

            if not is_gendered_entity(entity):
                entity_gender = None
            
            instructions[entity.id] = ReplaceInstruction(
                first_name=first_name,
                last_name=entity.last_name if not replace_last_names else last_name,
                gender=entity_gender,
                entity_category=entity_cat,
                is_modified=True
            )
        
        for entity in non_replaceable_entities:
            entity_gender = None
            if is_gendered_entity(entity):
                # The entity is only non replaceable if it's treated the same
                # across all categories
                entity_gender = random.choice(gender_list[0])

            last_name = entity.last_name if not replace_last_names else next(non_replaceable_last_names_iter)

            instructions[entity.id] = ReplaceInstruction(
                first_name=entity.first_name,
                last_name=entity.last_name if not replace_last_names else last_name,
                gender=entity_gender,
                entity_category=None,
                is_modified=replace_last_names
            )
        
        yield instructions


def generate_single_category_name_assignments(
        template: ReplaceTemplate,
        name_generator_factory: BBQNameGeneratorFactory,
        gender: NameGender,
        replace_last_names: bool,
        name_sources: dict[str, list[str]],
    ) -> list[dict[str, ReplaceInstruction]]:
    name_gen = name_generator_factory()
    gendered_entities = [e for e in template.replaceable_entities if is_gendered_entity(e)]
    if len(name_sources["male" if gender == NameGender.MALE else "female"]) != 1:
        raise ValueError("Mono only supports one category per gender")
    name_cat = name_sources["male" if gender == NameGender.MALE else "female"][0]

    first_names_iter = iter(name_gen.sample_first_names(gender, name_cat, len(gendered_entities)))
    if replace_last_names:
        last_names_iter = iter(name_gen.sample_last_names(name_cat, len(template.replaceable_entities)))

    instructions = {}
    for entity in template.replaceable_entities:
        last_name = next(last_names_iter) if replace_last_names else entity.last_name

        markers = {}
        first_name = None
        if is_gendered_entity(entity):
            markers["first_name_category"] = name_cat
            markers["gender"] = gender.value
            first_name = next(first_names_iter)

        if replace_last_names:
            markers["last_name_category"] = name_cat

        instructions[entity.id] = ReplaceInstruction(
            first_name=first_name,
            last_name=last_name,
            gender=gender,
            markers=markers
        )
    
    return instructions






def name_from_args(args):
    segments = ["onto-nw"]
    if args.preset is not None:
        segments.append(args.preset)
    
    if args.categories is not None:
        segments.append(",".join(args.categories))
    
    if args.name_sources is not None:
        segments.append(",".join("{}={}".format(k, "-".join(v)) for k, v in args.name_sources.items() if k != "unknown" or args.names == "first_last"))
    
    if args.category_genders is not None:
        segments.append(",".join(c + ":" +  "_".join("f" if g == NameGender.FEMALE else "m" for g in genders) for c, genders in args.category_genders.items()))

    if args.names is not None:
        segments.append(args.names)

    if args.no_pronouns:
        segments.append("no-pron")

    segments.append(args.mode)
    segments.append(str(args.samples))

    return "_".join(segments)

@dataclass
class GenerationSettings:
    categories: tuple[str, ...]
    name_sources: dict[str, tuple[str, ...]]
    category_genders: dict[str, tuple[NameGender, ...]]
    names: str

    @classmethod
    def from_args(cls, args) -> "GenerationSettings":
        if args.preset is not None:
            if args.preset not in PRESETS:
                raise ValueError("Invalid preset")
            val = PRESETS[args.preset]
        else:
            val = cls(None, None, None, None)
        
        if args.categories is not None:
            val.categories = args.categories
        
        if args.name_sources is not None:
            val.name_sources = args.name_sources
        
        if args.category_genders is not None:
            val.category_genders = args.category_genders

        if args.names is not None:
            val.names = args.names

        if val.categories is None or val.name_sources is None or val.category_genders is None or val.names is None:
            raise ValueError("Invalid settings")
        
        return val



PRESETS = {
    "gender": GenerationSettings(
        categories=("male", "female"),
        name_sources={"male": ("common",), "female": ("common",)},
        category_genders={"male": (NameGender.MALE,), "female": (NameGender.FEMALE,)},
        names="first"
    ),
    "race": GenerationSettings(
        categories=("black", "white"),
        name_sources={"black": ("black",), "white": ("white",)},
        category_genders={"black": (NameGender.FEMALE, NameGender.MALE), "white": (NameGender.FEMALE, NameGender.MALE)},
        names="first_last"
    )
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default=None)

    parser.add_argument("--categories", default=None, type=lambda x: x.split(","))
    parser.add_argument("--name-sources", default=None, type=parse_name_config)
    parser.add_argument("--category-genders", default=None, type=parse_gender_config)

    parser.add_argument("--names", type=str, choices=["first", "first_last"])
    parser.add_argument("--mode", type=str, default="balanced", choices=["balanced", "mono"])
    parser.add_argument("--out-path", type=Path, default=Path("instances"))
    parser.add_argument("--onto-path", type=Path, default=Path("/data2/summary_permutation_response/ontonotes-release-5.0/data/files/data/english/annotations/nw/"))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--no-pronouns", action="store_true", default=False)

    args = parser.parse_args()

    settings = GenerationSettings.from_args(args)

    base_path = args.onto_path
    name_generator = BBQNameGeneratorFactory()
    total = 0

    n_samples_per_article = args.samples

    all_samples = []

    for f in iter_onto_files(base_path):
        file_obj = OntoNoteFile.from_dir_and_name(*f)
        if file_obj is None:
            continue

        print("Processing", file_obj.name)

        template = ReplaceTemplate.from_onto_notes_file(file_obj)
        all_entities = template.replaceable_entities

        gendered_entities = []
        non_gendered_entities = []

        for entity in all_entities:
            if entity.first_name is not None or any(m.category in (MentionCategory.MR_MRS_MS, MentionCategory.PRONOUN_3P_INDIR, MentionCategory.PRONOUN_3P_NOM, MentionCategory.PRONOUN_3P_POSS, MentionCategory.PRONOUN_3P_REFL) for m in entity.mentions):
                gendered_entities.append(entity)
            else:
                non_gendered_entities.append(entity)

        if len(gendered_entities) == 0:
            continue

        try:
            for sample_idx in range(n_samples_per_article):
                if args.mode == "balanced":
                    for pair_id, instructions in enumerate(generate_paired_name_assignments(
                        template,
                        name_generator,
                        categories=settings.categories,
                        replace_last_names=settings.names == "first_last",
                        name_sources=settings.name_sources,
                        category_genders=settings.category_genders
                    )):
                        text = format_text(template.fill_with_entities(instructions, no_pronouns=args.no_pronouns))
                        all_samples.append(
                            {
                                "text": text,
                                "instructions": {k: i.asdict() for k, i in instructions.items()},
                                "sample_id": sample_idx,
                                "article_id": file_obj.name,
                                "pair_id": pair_id,
                            }
                        )
                elif args.mode == "mono":
                    for category in settings.categories:
                        instructions = next(generate_paired_name_assignments(
                            template,
                            name_generator,
                            categories=[category],
                            replace_last_names=settings.names == "first_last",
                            name_sources=settings.name_sources,
                            category_genders=settings.category_genders
                        ))
                        text = format_text(template.fill_with_entities(instructions, no_pronouns=args.no_pronouns))
                        all_samples.append(
                            {
                                "text": text,
                                "instructions": {k: i.asdict() for k, i in instructions.items()},
                                "sample_id": sample_idx,
                                "article_id": file_obj.name,
                                "text_category": category,
                            }
                        )
            total += 1
        except RuntimeError as e:
            if isinstance(e.__cause__, StopIteration):
                print("Skipping", file_obj.name, "due to insufficient name inventory")
            else:
                raise e

    
    name = name_from_args(args)
    
    print("Generated {} from {} articles".format(len(all_samples), total))

    args.out_path.mkdir(exist_ok=True, parents=True)
    fname = args.out_path / "{}.jsonl".format(name)
    with open(fname, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Wrote sample to {fname}")