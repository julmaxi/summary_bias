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

import math
from dataclasses import asdict

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

    if ";" in s:
        for part in s.split(";"):
            k, v = part.split("=")
            out[k] = v.split(",")
    else:
        out = {
            "male": [s],
            "female": [s],
            "unknown": [s]
        }
    
    return out


def is_gendered_entity(e):
    return e.first_name is not None or any(m.category in (MentionCategory.MR_MRS_MS, MentionCategory.PRONOUN_3P_INDIR, MentionCategory.PRONOUN_3P_NOM, MentionCategory.PRONOUN_3P_POSS, MentionCategory.PRONOUN_3P_REFL) for m in e.mentions)


def generate_gender_paired_name_assignments(
        template: ReplaceTemplate,
        name_generator_factory: BBQNameGeneratorFactory,
        replace_last_names: bool,
        name_sources: dict[str, list[str]],
    ) -> list[dict[str, ReplaceInstruction]]:
    gendered_entities = [e for e in template.replaceable_entities if is_gendered_entity(e)]

    half_gendered_entity_cnt = int(math.ceil(len(gendered_entities) / 2))
    
    gender_assignment = ["f"] * half_gendered_entity_cnt + ["m"] * half_gendered_entity_cnt
    random.shuffle(gender_assignment)

    names = name_generator_factory()

    male_first_names = [(cat, n) for cat in name_sources["male"] for n in names.sample_first_names(NameGender.MALE, cat, half_gendered_entity_cnt)]
    female_first_names = [(cat, n) for cat in name_sources["female"] for n in names.sample_first_names(NameGender.FEMALE, cat, half_gendered_entity_cnt)]

    if replace_last_names:
        male_last_names = [(cat, n) for cat in name_sources["male"] for n in names.sample_last_names(cat, half_gendered_entity_cnt)]
        random.shuffle(male_last_names)
        female_last_names = [(cat, n) for cat in name_sources["female"] for n in names.sample_last_names(cat, half_gendered_entity_cnt)]
        random.shuffle(female_last_names)
        non_gendered_last_names = [(cat, n) for cat in name_sources["unknown"] for n in names.sample_last_names(cat, len(template.replaceable_entities) - len(gendered_entities))]
        random.shuffle(non_gendered_last_names)

    for pair_idx in (NameGender.FEMALE, NameGender.MALE):
        instructions = {}

        male_first_names_iter = iter(male_first_names)
        female_first_names_iter = iter(female_first_names)

        if replace_last_names:
            male_last_names_iter = iter(male_last_names)
            female_last_names_iter = iter(female_last_names)
            non_gendered_last_names_iter = iter(non_gendered_last_names)

        if pair_idx == 1:
            gender_assignment_iter = iter(["m" if v == "f" else "f" for v in gender_assignment])
        else:
            gender_assignment_iter = iter(gender_assignment)

        for entity in template.replaceable_entities:
            entity_gender = None
            first_name = None
            if is_gendered_entity(entity):
                entity_gender = next(gender_assignment_iter)
                if entity_gender == "m":
                    entity_gender = NameGender.MALE
                else:
                    entity_gender = NameGender.FEMALE
            
                if entity_gender == NameGender.MALE:
                    first_name_cat, first_name = next(male_first_names_iter)
                else:
                    first_name_cat, first_name = next(female_first_names_iter)
            
            if replace_last_names:
                if entity_gender == NameGender.MALE:
                    last_name_cat, last_name = next(male_last_names_iter)
                elif entity_gender == NameGender.FEMALE:
                    last_name_cat, last_name = next(female_last_names_iter)
                else:
                    last_name_cat, last_name = next(non_gendered_last_names_iter)

            markers = {}
            if is_gendered_entity(entity):
                markers["first_name_category"] = first_name_cat
                markers["gender"] = entity_gender.value

            if replace_last_names:
                markers["last_name_category"] = last_name_cat
            
            #if is_gendered_entity(entity) or replace_last_names:
            instructions[entity.id] = ReplaceInstruction(
                first_name=first_name,
                last_name=entity.last_name if not replace_last_names else last_name,
                gender=entity_gender,
                markers=markers
            )
        
        yield instructions


def generate_single_gender_name_assignments(
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
    return "onto-nw_{gender}_{names}_{name_sources}_{samples}".format(
        gender=args.gender,
        names=args.names,
        name_sources=",".join("{}={}".format(k, "-".join(v)) for k, v in args.name_sources.items() if k != "unknown" or args.names == "first_last"),
        samples=args.samples
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--mode", type=str, default="paired")
    parser.add_argument("--gender", type=str, default="balanced", choices=["balanced", "mono"])
    parser.add_argument("--names", type=str, default="first", choices=["first", "first_last"])
    parser.add_argument("--name-sources", default={"male": ["common"], "female": ["common"], "unknown": ["common"]}, type=parse_name_config)
    parser.add_argument("--out-path", type=Path, default=Path("instances"))
    parser.add_argument("--onto-path", type=Path, default=Path("/data2/summary_permutation_response/ontonotes-release-5.0/data/files/data/english/annotations/nw/"))

    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    base_path = args.onto_path
    name_generator = BBQNameGeneratorFactory()
    total = 0

    n_samples_per_article = args.samples

    all_samples = []

    for f in iter_onto_files(base_path):
        file_obj = OntoNoteFile.from_dir_and_name(*f)
        print("Processing", file_obj.name)
        if file_obj is None:
            continue

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

        total += 1

        for sample_idx in range(n_samples_per_article):
            if args.gender == "balanced":
                for pair_id, instructions in enumerate(generate_gender_paired_name_assignments(template, name_generator, replace_last_names=args.names == "first_last", name_sources=args.name_sources)):
                    text = format_text(template.fill_with_entities(instructions))
                    all_samples.append(
                        {
                            "text": text,
                            "instructions": {k: i.asdict() for k, i in instructions.items()},
                            "sample_id": sample_idx,
                            "article_id": file_obj.name,
                            "pair_id": pair_id,
                        }
                    )
            elif args.gender == "mono":
                for gender in [NameGender.MALE, NameGender.FEMALE]:
                    instructions = generate_single_gender_name_assignments(template, name_generator, gender, replace_last_names=args.names == "first_last", name_sources=args.name_sources)
                    text = format_text(template.fill_with_entities(instructions))
                    all_samples.append(
                        {
                            "text": text,
                            "instructions": {k: i.asdict() for k, i in instructions.items()},
                            "sample_id": sample_idx,
                            "article_id": file_obj.name,
                            "text_gender": gender.value
                        }
                    )

    
    name = name_from_args(args)
    
    print("Generated {} from {} articles".format(len(all_samples), total))

    args.out_path.mkdir(exist_ok=True, parents=True)
    fname = args.out_path / "{}.jsonl".format(name)
    with open(fname, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Wrote sample to {fname}")