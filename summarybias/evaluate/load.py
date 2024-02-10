from dataclasses import dataclass
import json
from .parse import SummaryParse
from ..instancegen.templates import ReplaceInstruction

@dataclass
class Instance:
    original_article_id: str
    source: str
    summary: str
    instructions: dict[str, "ReplaceInstruction"]
    parse: dict[str, list[dict[str, str]]]
    text_category: str
    sample_id: int

    def from_dict(d: dict) -> "Instance":
        parse = SummaryParse.from_dict(d["parse"])
        parse.text = d["summary"]
        return Instance(
            original_article_id=d["article_id"],
            source=d["text"],
            summary=d["summary"],
            parse=parse,
            instructions={k: ReplaceInstruction.from_dict(v) for k, v in d["instructions"].items()},
            text_category=d.get("text_category"),
            sample_id=d["sample_id"]
        )
    
    @classmethod
    def from_jsonl(cls, path) -> list["Instance"]:
        with open(path, "r") as f:
            return [cls.from_dict(json.loads(line)) for line in f.readlines()]
