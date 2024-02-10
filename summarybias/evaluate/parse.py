import spacy
import argparse
import json
import tqdm
from pathlib import Path

from dataclasses import dataclass, asdict

@dataclass
class NamedEntity:
    start: int
    end: int
    label: str

    @property
    def text(self):
        end_char = self.parse.tokens[self.end - 1][1]
        return self.parse.text[self.parse.tokens[self.start][0]:end_char]
    
    def __iter__(self):
        for i in range(self.start, self.end):
            yield self.parse.get_token(i)

@dataclass
class SummaryParse:
    sentences: list[tuple[int, int]]
    tokens: list[tuple[int, int]]
    named_entities: list[NamedEntity]

    def __init__(self, sentences, tokens, named_entities):
        self.sentences = sentences
        self.tokens = tokens
        self.named_entities = named_entities

        for n in self.named_entities:
            n.parse = self

    def from_spacy(doc: spacy.tokens.Doc):
        try:
            sentences = [(sent.start, sent.end) for sent in doc.sents]
        except:
            sentences = [(0, len(doc))]
        tokens = [(token.idx, token.idx + len(token.text)) for token in doc]
        named_entities = [NamedEntity(ent.start, ent.end, ent.label_) for ent in doc.ents]
        return SummaryParse(sentences, tokens, named_entities)
    
    def get_token(self, idx: int):
        return self.text[self.tokens[idx][0]:self.tokens[idx][1]]

    def __iter__(self):
        for i in range(len(self.tokens)):
            yield self.get_token(i)
    
    @classmethod
    def from_dict(cls, d: dict) -> "SummaryParse":
        return SummaryParse(
            sentences=d["sentences"],
            tokens=d["tokens"],
            named_entities=[NamedEntity(**ne) for ne in d["named_entities"]]
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("-d", "--dir", dest="out_dir", type=Path, default="parsed_summaries")
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_trf")

    with open(args.input_file, "r") as f, open(args.out_dir / args.input_file.name, "w") as out:
        items = [json.loads(line) for line in f.readlines()]
    
        for item in tqdm.tqdm(items):
            doc = nlp(item["summary"])
            parse = SummaryParse.from_spacy(doc)
            item["parse"] = asdict(parse)
            out.write(json.dumps(item) + "\n")
