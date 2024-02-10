from typing import Any
import pandas as pd

from dataclasses import dataclass
from collections import defaultdict
import itertools as it

import random
import enum

from pathlib import Path


class NameGender(enum.Enum):
    FEMALE = "female"
    MALE = "male"

class BBQNameGeneratorFactory:
    def __init__(self, bbq_path: str = Path("data/vocabulary_proper_names.csv"), census_extend_path=Path("data/"), surnames_path=Path("data/Names_2010Census.csv")) -> None:
        self.bbq_path = bbq_path
        name_df = pd.read_csv(bbq_path)
        self.first_names = defaultdict(lambda: defaultdict(list))
        self.last_names = defaultdict(list)

        for row in name_df.itertuples():
            if row.First_last == "last":
                self.last_names[row.ethnicity.lower()].append(row.Name)
            else:
                gender = row.gender.lower()

                category = None
                if row.First_last == "first_only":
                    category = "common"
                else:
                    category = row.ethnicity.lower()
                
                if gender == "m":
                    self.first_names[category][NameGender.MALE].append(row.Name)
                else:
                    self.first_names[category][NameGender.FEMALE].append(row.Name)
        
        if census_extend_path is not None:
            def read(path):
                lines = path.read_text().splitlines()
                return [l.split()[0].capitalize() for l in lines]
            male_added = read(census_extend_path / "dist.male.first")
            female_added = read(census_extend_path / "dist.female.first")
            
            self.first_names[category][NameGender.MALE].extend(male_added[20:50])
            self.first_names[category][NameGender.FEMALE].extend(female_added[20:50])

        with open(
            surnames_path, "r", encoding="utf-8"
        ) as f:
            next(f) # skip header
            names = [line.strip().split(",")[0].capitalize() for line, _ in zip(f, range(1000))]
            self.last_names["common"] = names
            
    
    def __call__(self) -> "BBQNameGenerator":
        return BBQNameGenerator(self)


class BBQNameGenerator:
    def __init__(self, factory) -> None:
        self.iterators = {}
        self.factory = factory
    
    def next_first_name(self, gender: NameGender, category: str) -> str:
        if (category, gender) not in self.iterators:
            vals = list(
                self.factory.first_names[category][gender]
            )
            random.shuffle(vals)
            self.iterators[category, gender] = iter(vals)

        return next(self.iterators[category, gender])
    
    def sample_first_names(self, gender: NameGender, category: str, n: int) -> list[str]:
        out = []
        for _ in range(n):
            out.append(self.next_first_name(gender, category))
        return out
    
    def next_last_name(self, category: str) -> str:
        if category not in self.iterators:
            vals = list(
                self.factory.last_names[category]
            )
            random.shuffle(vals)
            self.iterators[category] = iter(vals)

        return next(self.iterators[category])
    
    def sample_last_names(self, category: str, n: int) -> list[str]:
        out = []
        for _ in range(n):
            out.append(self.next_last_name(category))
        return out
        




if __name__ == "__main__":
    generator = BBQNameGeneratorFactory()

    print(generator.generate_gender_balanced_first_names(10))
    