from pathlib import Path

class ListBasedNameGenderClassifier:
    @classmethod
    def from_census_data(cls, base_path: Path = Path("./data/")) -> "ListBasedNameGenderClassifier":
        male_path = base_path / "dist.male.first"
        female_path = base_path / "dist.female.first"

        def read_file(path):
            return [
                (l.split()[0].lower(), float(l.split()[1])) for l in path.read_text().splitlines()
            ]
        
        male_names_and_freqs = dict(read_file(male_path))
        female_names_and_freqs = dict(read_file(female_path))

        for name, freq in list(male_names_and_freqs.items()):
            f_freq = female_names_and_freqs.get(name)
            if f_freq is not None:
                if freq / f_freq > 2:
                    del female_names_and_freqs[name]
                elif freq / f_freq < 0.5:
                    del male_names_and_freqs[name]
                else:
                    del female_names_and_freqs[name]
                    del male_names_and_freqs[name]
        
        wiki_path = base_path / "wikipedia_entities.csv"

        ne_genders = {}

        with open(wiki_path, "r", encoding="utf-8") as f:
            next(f) # skip header
            for line in f:
                name, gender = line.strip().split(",", 1)
                if len(name.split()) == 1:
                    continue
                ne_genders[name.lower()] = gender

        return cls(
            {
                "male": list(male_names_and_freqs.keys()),
                "female": list(female_names_and_freqs.keys())
            },
            ne_genders
        )

    def __init__(self, category_word_lists, named_entities) -> None:
        self.named_entities = named_entities
        self.category_word_lists = category_word_lists

    def __call__(self, tokens) -> str:
        out_cat = None

        ne_gender = self.named_entities.get(" ".join(tokens).lower())
        if ne_gender is not None:
            if ne_gender == "m":
                return "male"
            elif ne_gender == "f":
                return "female"

        for cat, word_list in self.category_word_lists.items():
            if any(tok.lower() in word_list for tok in tokens):
                if out_cat is None:
                    out_cat = cat
                else:
                    return None

        return out_cat
    
    @classmethod
    def get_supported_categories(cls) -> list[str]:
        return ["male", "female"]
