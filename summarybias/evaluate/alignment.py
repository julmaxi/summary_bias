from ..instancegen.templates import ReplaceInstruction
from .parse import NamedEntity, SummaryParse
from .load import Instance
from dataclasses import dataclass


@dataclass
class AlignedInstance:
    original_article_id: str
    parse: SummaryParse
    instructions: dict[str, ReplaceInstruction]
    source_to_summary_alignment: dict[str, list[int]]
    summary_to_source_alignment: dict[int, list[str]]
    text_category: str
    source: str
    sample_id: int

    @classmethod
    def from_instance(cls, instance: "Instance") -> "AlignedInstance":
        source_to_summary_alignment, summary_to_source_alignment = align_entities(
            instance.parse.named_entities, instance.instructions
        )

        return AlignedInstance(
            original_article_id=instance.original_article_id,
            parse=instance.parse,
            instructions=instance.instructions,
            source_to_summary_alignment=source_to_summary_alignment,
            summary_to_source_alignment=summary_to_source_alignment,
            text_category=instance.text_category,
            source=instance.source,
            sample_id=instance.sample_id,
        )


def check_name_part_is_valid(name_token: str, entity: ReplaceInstruction) -> bool:
    name_token_lc = name_token.lower()
    EXCLUSION_LIST = "Mr. Mrs. Ms. Sir".lower().split()
    return (
        name_token_lc in EXCLUSION_LIST
        or name_token_lc in [n.lower() for n in [entity.first_name, entity.last_name] if n is not None]
    )

def align_entities(entities: list[NamedEntity], instructions: dict[str, ReplaceInstruction]) -> tuple[dict, dict]:
    entities = [(idx, e) for idx, e in enumerate(entities) if e.label == "PERSON"]
    entity_map = {k: [] for k in instructions.keys()}
    reverse_entity_map = {idx: [] for idx, _ in entities}

    for summary_entity_id, summary_entity in entities:
        name_parts = list(e for e in summary_entity if e.isalpha())
        for source_entity_id, source_entity in instructions.items():
            if (source_entity.last_name is not None) and source_entity.last_name.lower() in summary_entity.text.lower() or (source_entity.first_name is not None and len(list(summary_entity)) == 1 and source_entity.first_name.lower() in summary_entity.text.lower()):
                if not all(
                    check_name_part_is_valid(name_part, source_entity) for name_part in name_parts
                ):
                    continue
                entity_map[source_entity_id].append(summary_entity_id)
                reverse_entity_map[summary_entity_id].append(source_entity_id)
        
    return entity_map, reverse_entity_map
