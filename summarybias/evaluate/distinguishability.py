from .measures import ParsedInstance, ConfidenceInterval, ReplaceInstruction
from typing import Any, Callable, Protocol
import numpy as np
import itertools as it
import re
import scipy
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

from sentence_transformers import SentenceTransformer
import sys

class CategorizedInstance(Protocol):
    @property
    def text_category(self) -> str:
        pass

    @property
    def original_article_id(self) -> str:
        pass

    @property
    def summary(self) -> str:
        pass

    @property
    def instructions(self) -> dict[str, ReplaceInstruction]:
        pass


class TransformerEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.model.encode([t for t in texts])

class BoWEmbedder:
    def __call__(self, texts: list[str]) -> np.ndarray:
        return CountVectorizer(stop_words="english").fit_transform(t for t in texts).toarray()


def anonymize_summary(instance: CategorizedInstance) -> str:
    summary = instance.summary

    for _, instruction in instance.instructions.items():
        if instruction.first_name:
            summary = summary.replace(instruction.first_name, "FIRST_NAME")
        summary = summary.replace(instruction.last_name, "LAST_NAME")

    BASE_REPLACEMENT = {
        "he": "they",
        "she": "they",
        "his": "their",
        "her": "their",
        "him": "their",
        "hers": "theirs",
        "himself": "themself",
        "herself": "themself",
        "himselves": "themself",
        "herselves": "themself",
        "mr": "M.",
        "mrs": "M.",
        "ms": "M.",
        "miss": "M.",
        "sir": "M.",
        "ma'am": "M.",
        "sirs": "M.",
        "madams": "M.",
    }

    REPLACEMENT = {
    }

    for k, v in BASE_REPLACEMENT.items():
        REPLACEMENT[k.capitalize()] = v.capitalize()
        REPLACEMENT[k] = v

    for k, v in REPLACEMENT.items():
        summary = re.sub(r"\b{}\b".format(k), v, summary)

    return summary

def cats_to_ints(cats: list[str]) -> np.ndarray:
    cat_to_int = {cat: idx for idx, cat in enumerate(sorted(set(cats)))}
    return np.array([cat_to_int[cat] for cat in cats])

def compute_distinguishability_with_ci(
    instances: list[CategorizedInstance],
    embedding_func: Callable[[list[str]], np.ndarray],
    n_bootrap_samples: int = 1000,
) -> tuple[float, ConfidenceInterval]:
    instances = sorted(instances, key=lambda i: (i.original_article_id, i.text_category))

    per_original_scores = []

    for _, group in it.groupby(instances, key=lambda i: i.original_article_id):
        group = list(group)
        embeddings = embedding_func([anonymize_summary(i) for i in group])
        similarities = 1.0 - scipy.spatial.distance.pdist(embeddings, metric="cosine")
        sim_matrix = scipy.spatial.distance.squareform(similarities)
        cat_ids = cats_to_ints([i.text_category for i in group])

        n_trial = 0
        n_succ = 0

        #if len(cat_counts) == 1 or min(cat_counts.values()) == 1:
        #    continue
        #print(sim_matrix)
        for instance_idx, cat_id in enumerate(cat_ids):
            #if len(set(cat_ids)) == 2 and cat_id == 1:
            #    continue
            same_cat_mask = cat_ids == cat_id
            same_class_sim_sum = sim_matrix[instance_idx, same_cat_mask].sum() - sim_matrix[instance_idx, instance_idx]
            same_class_mean_sim = same_class_sim_sum / (same_cat_mask.sum() - 1)
            diff_class_mean_sim = sim_matrix[instance_idx, ~same_cat_mask].mean()

            #print(same_class_mean_sim, diff_class_mean_sim)
            if same_class_mean_sim > diff_class_mean_sim:
                n_succ += 1
            elif same_class_mean_sim == diff_class_mean_sim:
                n_succ += 0.5
            n_trial += 1

        #print(n_succ, n_trial)
        
        per_original_scores.append(((n_succ / n_trial) - 0.5) / 0.5)
    
    score = np.mean(per_original_scores)

    if n_bootrap_samples is not None:
        ci = ConfidenceInterval.from_article_scores(
            per_original_scores,
            n_bootrap_samples=n_bootrap_samples,
        )
    else:
        ci = None
    
    return score, ci


