from typing import Generator
import xml.etree.ElementTree as ET
from pathlib import Path

from dataclasses import dataclass
import re
import xml.parsers.expat
from collections import defaultdict


@dataclass
class OntoNoteCorefEntry:
    start_tok: int
    end_tok: int

    nesting_level: int
    type_: str
    sub_type: str


@dataclass
class OntoNoteNamedEntity:
    start_tok: int
    end_tok: int

    entity_type: str

@dataclass
class OntoNoteFile:
    name: str
    toks: list[str]
    token_starts: list[int]
    coref_chains: dict[str, list[OntoNoteCorefEntry]]
    named_entities: list[OntoNoteNamedEntity]
    pos_tags: list[str]


    @classmethod
    def from_dir_and_name(
        self,
        dirname: str,
        name: str
    ) -> "OntoNoteFile":
        dirname = Path(dirname)
        coref_file = dirname / (name + ".coref")
        
        if not coref_file.exists():
            return None

        coref_parse = XMLParse.from_file(coref_file.open("r"))
        name_parse = XMLParse.from_file((dirname / (name + ".name")).open("r"))
        pos_file = OntonotesPOSFile.from_file((dirname / (name + ".parse")).open("r"))
        pos_tag_toks, pos_tags = pos_file.to_clean_tagged_toks()

        assert len(coref_parse.root.children) == 1
        assert coref_parse.toks == name_parse.toks == pos_tag_toks, f"{name} {len(coref_parse.toks)} {len(name_parse.toks)} {len(pos_tag_toks)}"
        chains = self.coref_xml_to_chains(coref_parse)

        named_entities = [OntoNoteNamedEntity(
            start_tok=e.start,
            end_tok=e.end,
            entity_type=e.attrs["TYPE"]
        ) for e in name_parse.iter_all_tags() if e.name == "ENAMEX"]

        token_starts = [0]
        for tok in coref_parse.toks:
            token_starts.append(token_starts[-1] + len(tok) + 1)
        token_starts = token_starts[:-1]

        return OntoNoteFile(
            name=name,
            toks=coref_parse.toks,
            token_starts=token_starts,
            coref_chains=chains,
            named_entities=named_entities,
            pos_tags=pos_tags
        )

    @staticmethod
    def coref_xml_to_chains(coref_parse: "XMLParse") -> dict[str, list[OntoNoteCorefEntry]]:
        real_root = coref_parse.root.children[0]

        chains = defaultdict(list)

        stack = [(real_root, 0)]

        while True:
            if len(stack) == 0:
                break
            curr, coref_nest_level = stack.pop()

            if curr.name == "COREF":
                chains[curr.attrs["ID"]].append(OntoNoteCorefEntry(
                    start_tok=curr.start,
                    end_tok=curr.end,
                    nesting_level=coref_nest_level,
                    type_=curr.attrs["TYPE"],
                    sub_type=curr.attrs.get("SUBTYPE")
                ))
                stack.extend([(child, coref_nest_level + 1) for child in reversed(curr.children)])
            else:
                stack.extend([(child, coref_nest_level) for child in reversed(curr.children)])

        return chains

@dataclass
class XMLTag:
    start: int
    end: int
    name: str
    attrs: dict

    children: list["XMLTag"]

@dataclass
class XMLParse:
    toks: list[str]
    root: XMLTag

    @classmethod
    def from_file(cls, f) -> "XMLParse":
        root = None
        curr_path = []
        all_toks = []

        accumulated_chars = []


        def chars_to_new_toks(chars):
            toks = chars.strip().split()
            
            return [tok for tok in toks if not re.match(r"(0|(\*PRO\*|\*U\*|\*T\*|\*)(-[0-9]+)?)", tok)]

        def start_element(name, attrs):
            nonlocal root, curr_path, accumulated_chars

            if len(accumulated_chars) > 0:
                all_toks.extend(chars_to_new_toks("".join(accumulated_chars)))

                accumulated_chars = []
            
            tag = XMLTag(len(all_toks), None, name, attrs, [])
            if root is None:
                root = tag
            else:
                curr_path[-1].children.append(tag)
            
            curr_path.append(tag)
        
        def end_element(name):
            nonlocal curr_path, accumulated_chars
            if len(accumulated_chars) > 0:
                all_toks.extend(chars_to_new_toks("".join(accumulated_chars)))
                accumulated_chars = []
            curr_path.pop().end = len(all_toks)
        
        def char_data(data):
            accumulated_chars.extend(data)

        parser = xml.parsers.expat.ParserCreate()
        parser.StartElementHandler = start_element
        parser.EndElementHandler = end_element
        parser.CharacterDataHandler = char_data
        parser.Parse(f.read(), True)

        return cls(all_toks, root)
    
    def iter_all_tags(self) -> Generator[XMLTag, None, None]:
        stack = [self.root]
        while True:
            if len(stack) == 0:
                break
            curr = stack.pop()
            yield curr
            stack.extend(curr.children)


@dataclass
class OntonotesPOSFile:
    pos_tagged_sents: list[list[(str, str)]]

    @classmethod
    def from_file(cls, f) -> "OntonotesPOSFile":
        pos_tagged_sents = []
        for sent_parse in f.read().split("\n\n"):
            sent_parse = sent_parse.strip()
            pos_tags = re.finditer(r"\(([^\s)(]+) ([^\s)()]+)\)", sent_parse)
            pos_tagged_words = []
            for match in pos_tags:
                pos_tagged_words.append((match.group(1), match.group(2)))
            pos_tagged_sents.append(pos_tagged_words)
        return cls(pos_tagged_sents)
    
    def to_clean_tagged_toks(self) -> tuple[list[str], list[str]]:
        toks = []
        tags = []
        for sent in self.pos_tagged_sents:
            for tag, word in sent:
                word = word.replace("&", "-AMP-")
                if re.match(r"\*PRO\*|\*U\*|\*T\*|\*|0", word):
                    continue
                toks.append(word)
                tags.append(tag)
        return toks, tags


def iter_onto_files(base_dir: Path) -> Generator[tuple[Path, str], None, None]:
    for path in base_dir.glob("**/*.coref"):
        yield (path.parent, path.name.rsplit(".")[0])

if __name__ == "__main__":
    # Tests ontonotes parsing
    import sys
    base_path = Path(sys.argv[1])
    for f in iter_onto_files(base_path):
        file_obj = OntoNoteFile.from_dir_and_name(*f)
        if file_obj is None:
            continue
        print(file_obj.coref_chains)



