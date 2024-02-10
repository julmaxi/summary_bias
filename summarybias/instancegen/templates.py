from .ontonotes import OntoNoteFile, OntoNoteNamedEntity, iter_onto_files, OntoNoteCorefEntry
from pathlib import Path

from dataclasses import dataclass, asdict
from enum import Enum
import json
from collections import defaultdict, Counter

from .names import NameGender

@dataclass
class ReplaceableEntity:
    id: str
    entity_type: str
    mentions: list["ReplaceableMention"]

    first_name: str = None
    last_name: str = None

@dataclass
class ReplaceableMention:
    start: int
    end: int
    category: "MentionCategory"
    original_text: str

class MentionCategory(Enum):
    FULL_NAME = 0
    FIRST_NAME = 1
    LAST_NAME = 2
    PRONOUN_3P_NOM = 3
    PRONOUN_3P_POSS = 4
    PRONOUN_3P_INDIR = 5
    MR_MRS_MS = 6
    PRONOUN_3P_REFL = 7
    SIR = 8
    OTHER = 9

def classify_mention(mention: str, tag: str) -> MentionCategory:
    if mention.lower() in ["mr.", "mrs.", "ms."]:
        return MentionCategory.MR_MRS_MS
    elif mention.lower() in ["sir", "lady"]:
        return MentionCategory.SIR
    if mention.lower() in ["he", "she"]:
        return MentionCategory.PRONOUN_3P_NOM
    elif mention.lower() in ["his", "her"] and tag == "PRP$":
        return MentionCategory.PRONOUN_3P_POSS
    elif mention.lower() in ["him", "her"] and tag == "PRP":
        return MentionCategory.PRONOUN_3P_INDIR
    elif mention.lower() in ["himself", "herself"] and tag == "PRP":
        return MentionCategory.PRONOUN_3P_REFL
    else:
        return None
    
def take(iter, n):
    for _ in range(n):
        next(iter)


def groupby(d, key):
    out = defaultdict(list)
    for k in d:
        out[key(k)].append(k)
    return out

@dataclass
class ReplaceTemplate:
    text: str
    replaceable_entities: list["ReplaceableEntity"]

    def fill_with_entities(self, replace_map, no_pronouns: bool = False) -> str:
        all_mentions = [(mention.start, mention.end, entity_idx, entity, mention) for (entity_idx, entity) in enumerate(self.replaceable_entities) for mention in entity.mentions]
        all_mentions.sort(reverse=True)

        if no_pronouns:
            all_mentions = [m for m in all_mentions if m[4].category not in [MentionCategory.PRONOUN_3P_NOM, MentionCategory.PRONOUN_3P_POSS, MentionCategory.PRONOUN_3P_INDIR, MentionCategory.PRONOUN_3P_REFL]]
    
        out = []
        text_iter = iter(self.text)
        next_idx = 0
        while next_idx < len(self.text):
            if len(all_mentions) == 0:
                break
            next_mention_start, next_mention_stop, _, next_entity,  next_mention = all_mentions[-1]
            while next_mention_start < next_idx:
                all_mentions.pop()
                if len(all_mentions) == 0:
                    break
                print(f"Warning, overlapping mentions {next_mention_start} {next_idx} {next_mention}")
                next_mention_start, next_mention_stop, _, next_entity,  next_mention = all_mentions[-1]
            
            if next_entity.id not in replace_map:
                all_mentions.pop()
                continue

            if len(all_mentions) == 0:
                break

            if next_mention_start == next_idx:
                all_mentions.pop()
                take(text_iter, next_mention_stop - next_mention_start)
                out.append(replace_map[next_entity.id].realization_for_category(category=next_mention.category, original_text=next_mention.original_text))
                next_idx = next_mention_stop
            else:
                out.append(next(text_iter))
                next_idx += 1

        out.extend(text_iter)
        return "".join(out)

    def show_with_placeholders(self) -> str:
        return self.fill_with_entities({e.id: PlaceholderFiller(e.id) for e in self.replaceable_entities})
    
    @staticmethod
    def chain_to_replaceable_entity(onto_file: OntoNoteFile, chain: list[tuple[int, int, bool]], id_: str) -> ReplaceableEntity:
        """
        Converts a chain to a replaceable entity. This is a helper function for from_onto_notes_file.
        We expect the chain to be cleaned before, i.e. this function does no checking for relevance.
        """
        toks = onto_file.toks

        last_name_candidates = []
        has_mr_mrs_ms = False
        first_name_candidates = []
        
        for (mention_start, mention_end, is_ne) in chain:
            if not is_ne:
                continue

            name_toks = toks[mention_start:mention_end]

            name_toks = [n for n in name_toks if n.lower() != "jr."]

            name_pos_tags = onto_file.pos_tags[mention_start:mention_end]
            name_toks = [t for t, p in zip(name_toks, name_pos_tags) if p == "NNP"]

            if mention_start > 0:
                prev_tok = toks[mention_start - 1]
                if prev_tok == "Mr." or prev_tok == "Mrs." or prev_tok == "Ms.":
                    if len(name_toks) == 1:
                        has_mr_mrs_ms = True
                        last_name_candidates = [name_toks[0]]
                        continue
            if len(name_toks) == 1 and not has_mr_mrs_ms:
                last_name_candidates.append(name_toks[0])
            else:
                if len(name_toks) > 0:
                    first_name_candidates.extend(name_toks[:-1])
                    last_name_candidates.append(name_toks[-1])
        
        last_name = None
        first_name = None

        if len(last_name_candidates) > 0:
            counts = Counter(c.lower() for c in last_name_candidates)
            last_name = counts.most_common(1)[0][0]
            last_name = last_name.capitalize()
        
        if len(first_name_candidates) > 0:
            counts = Counter(c.lower() for c in first_name_candidates)
            first_name = counts.most_common(1)[0][0]
            first_name = first_name.capitalize()
        
        replacable_mentions = []

        def process_token_class_buffer(buffer):
            if MentionCategory.FIRST_NAME in buffer and MentionCategory.LAST_NAME in buffer:
                mention_class = MentionCategory.FULL_NAME
            elif MentionCategory.FIRST_NAME in buffer:
                mention_class = MentionCategory.FIRST_NAME
            elif MentionCategory.LAST_NAME in buffer:
                mention_class = MentionCategory.LAST_NAME
            else:
                print("Unclassifiable", buffer)
                mention_class = None
            return mention_class

        for (mention_start, mention_end, is_ne) in chain:
            tok_class_buffer = []
            tok_buffer_start = None
            for tok in range(mention_start, mention_end):
                mention_class = classify_mention(toks[tok], onto_file.pos_tags[tok])

                if mention_class is None:
                    if first_name is not None and toks[tok].lower() == first_name.lower():
                        mention_class = MentionCategory.FIRST_NAME
                    elif last_name is not None and toks[tok].lower() == last_name.lower():
                        mention_class = MentionCategory.LAST_NAME
                    elif onto_file.pos_tags[tok] == "NNP" and is_ne:
                        # Deal with inconsistent last names (see LOG)
                        if first_name is not None:
                            mention_class = MentionCategory.FIRST_NAME
                        else:
                            mention_class = MentionCategory.LAST_NAME
                
                if mention_class in [MentionCategory.FIRST_NAME, MentionCategory.LAST_NAME]:
                    if len(tok_class_buffer) == 0:
                        tok_buffer_start = tok
                    tok_class_buffer.append(mention_class)
                else:
                    if mention_class is not None:
                        replacable_mentions.append(
                            ReplaceableMention(
                                start=onto_file.token_starts[tok],
                                end=onto_file.token_starts[tok] + len(onto_file.toks[tok]),
                                category=mention_class,
                                original_text=toks[tok]
                            )
                        )
                    if len(tok_class_buffer) > 0:
                        buffer_class = process_token_class_buffer(tok_class_buffer)
                        if buffer_class is not None:
                            replacable_mentions.append(
                                ReplaceableMention(
                                    start=onto_file.token_starts[tok_buffer_start],
                                    end=onto_file.token_starts[tok - 1] + len(onto_file.toks[tok - 1]),
                                    category=buffer_class,
                                    original_text=" ".join(toks[tok_buffer_start:tok])
                                )
                            )
                        tok_buffer_start = None
                        tok_class_buffer.clear()
                    
            if len(tok_class_buffer) > 0:
                buffer_class = process_token_class_buffer(tok_class_buffer)
                if buffer_class is not None:
                    replacable_mentions.append(
                        ReplaceableMention(
                            start=onto_file.token_starts[tok_buffer_start],
                            end=onto_file.token_starts[tok] + len(onto_file.toks[tok]),
                            category=buffer_class,
                            original_text=" ".join(toks[tok_buffer_start:tok + 1])
                        )
                    )
        replacable_mentions.sort(key=lambda m: m.start)
        return ReplaceableEntity(
            id=id_,
            entity_type="PERSON",
            mentions=replacable_mentions,
            first_name=first_name,
            last_name=last_name
        )


    @classmethod
    def from_onto_notes_file(cls, onto_file: "OntoNoteFile") -> "ReplaceTemplate":
        chain_candidate_named_entities : dict[str, list[OntoNoteNamedEntity]] = {}
        for chain_id, chain in onto_file.coref_chains.items():
            chain_candidate_named_entities[chain_id] = [[] for _ in range(len(chain))]
        relevant_named_entities = [e for e in onto_file.named_entities if e.entity_type == "PERSON"]
        
        named_entity_chains = [[] for _ in range(len(relevant_named_entities))]

        for chain_id, chain in onto_file.coref_chains.items():
            for mention_idx, mention in enumerate(chain):
                for entity_idx, entity in enumerate(relevant_named_entities):
                    if mention.start_tok <= entity.start_tok and mention.end_tok >= entity.end_tok:
                        chain_candidate_named_entities[chain_id][mention_idx].append(entity)
                        named_entity_chains[entity_idx].append((chain_id, mention_idx, mention))

        
        named_entity_linked_mentions = {}
        for named_entity_id, candidate_chains in enumerate(named_entity_chains):
            if len(candidate_chains) == 0:
                continue

            final_chain = None
            for chain_id, mention_idx, mention in sorted(candidate_chains, key=lambda x: x[2].nesting_level, reverse=True):
                if mention.type_ == "IDENT":
                    final_chain = chain_id, mention_idx
                    break
                else:
                    if mention.type_ == "APPOS" and mention.sub_type != "HEAD":
                        break
            
            if final_chain is None:
                continue
            
            if (chain_id, mention_idx) not in named_entity_linked_mentions:
                named_entity_linked_mentions[final_chain] = named_entity_id
            else:
                named_entity_linked_mentions[final_chain] = "__conflict__"
        
        named_entities_with_chain = set(
            named_entity_id for (chain_id, _), named_entity_id in named_entity_linked_mentions.items() if named_entity_id != "__conflict__"
        )

        named_entity_linked_mentions = {k: v for k, v in named_entity_linked_mentions.items() if v != "__conflict__"}

        relevant_chains = {}

        for named_entity_id, entity in enumerate(relevant_named_entities):
            if named_entity_id not in named_entities_with_chain:
                new_chain_id = f"NE_{named_entity_id}"
                relevant_chains[new_chain_id] = [(entity.start_tok, entity.end_tok, True)]
        
        mapped_mentions_by_chain = groupby(named_entity_linked_mentions.items(), lambda x: x[0][0])

        for chain_id, ne_members in mapped_mentions_by_chain.items():
            mentions_with_nes = {
                mention_idx: named_entity_id for (_chain_id, mention_idx), named_entity_id in ne_members
            }
            out_mentions = []
            for mention_idx, mention in enumerate(onto_file.coref_chains[chain_id]):
                if mention_idx in mentions_with_nes:
                    ne = relevant_named_entities[mentions_with_nes[mention_idx]]
                    if mention.start_tok < ne.start_tok:
                        out_mentions.append((mention.start_tok, ne.start_tok, False))

                    out_mentions.append((ne.start_tok, ne.end_tok, True))

                    if ne.end_tok < mention.end_tok:
                        out_mentions.append((ne.end_tok, mention.end_tok, False))
                else:
                    out_mentions.append((mention.start_tok, mention.end_tok, False))
            
            out_mentions.sort(key=lambda x: x[0])
            relevant_chains[chain_id] = out_mentions

        replaceable_entities = []
        for chain_id, chain in relevant_chains.items():
            entity = ReplaceTemplate.chain_to_replaceable_entity(onto_file, chain, chain_id)
            if len(entity.mentions) > 0:
                replaceable_entities.append(entity)
        return ReplaceTemplate(
            text=" ".join(onto_file.toks),
            replaceable_entities=replaceable_entities
        )


@dataclass
class IdentityReplaceInstruction:
    def realization_for_category(self, original_text, **kwargs):
        return original_text


@dataclass
class ReplaceInstruction:
    first_name: str
    last_name: str
    gender: NameGender

    entity_category: str
    is_modified: bool

    def realization_for_category(self, category: MentionCategory, original_text, **kwargs):
        def get_val():
            if category == MentionCategory.FULL_NAME:
                return f"{self.first_name} {self.last_name}"
            elif category == MentionCategory.FIRST_NAME:
                return self.first_name
            elif category == MentionCategory.LAST_NAME:
                return self.last_name
            elif category == MentionCategory.PRONOUN_3P_NOM:
                if self.gender == NameGender.MALE:
                    return "he"
                else:
                    return "she"
            elif category == MentionCategory.PRONOUN_3P_INDIR:
                if self.gender == NameGender.MALE:
                    return "him"
                else:
                    return "her"
            elif category == MentionCategory.PRONOUN_3P_POSS:
                if self.gender == NameGender.MALE:
                    return "his"
                else:
                    return "her"
            elif category == MentionCategory.PRONOUN_3P_REFL:
                if self.gender == NameGender.MALE:
                    return "himself"
                else:
                    return "herself"
            elif category == MentionCategory.MR_MRS_MS:
                if self.gender == NameGender.MALE:
                    return "Mr."
                else:
                    return "Mrs."
            elif category == MentionCategory.SIR:
                if self.gender == NameGender.MALE:
                    return "Sir"
                else:
                    return "Lady"
            else:
                raise ValueError(f"Unknown category {category}")
            
        val = get_val()

        if original_text[0].isupper():
            val = val[0].upper() + val[1:]
        else:
            val = val.lower()
        
        return val

    def asdict(self):
        d = asdict(self)
        d["gender"] = d["gender"].value if d["gender"] is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ReplaceInstruction":
        return cls(
            first_name=d.get("first_name"),
            last_name=d.get("last_name"),
            gender=cls.parse_gender(d.get("gender")),
            entity_category=d.get("entity_category"),
            is_modified=d.get("is_modified")
        )

    @classmethod
    def parse_gender(cls, s: str) -> NameGender:
        if s is None:
            return None
        
        if s == "female":
            return NameGender.FEMALE
        elif s == "male":
            return NameGender.MALE
        
        raise ValueError(f"Invalid gender {s}")
    
        


class PlaceholderFiller:
    def __init__(self, id) -> None:
        self.id = id
    def realization_for_category(self, category, **kwargs):
        return f"__[{self.id}, {category}]__"


if __name__ == "__main__":
    base_path = Path("/data2/summary_permutation_response/ontonotes-release-5.0/data/files/data/english/annotations/nw/wsj")
    for f in iter_onto_files(base_path):
        file_obj = OntoNoteFile.from_dir_and_name(*f)
        if file_obj is None:
            continue
        #if f[1] != "wsj_1178":
        #if f[1] != "chtb_0302":
        #    continue

        template = ReplaceTemplate.from_onto_notes_file(file_obj)
        print(template.show_with_placeholders())
        for entity in template.replaceable_entities:
            print(entity.first_name, entity.last_name)
        print(f)
        print()
