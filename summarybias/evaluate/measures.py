from .parse import SummaryParse
from ..instancegen.templates import ReplaceInstruction
from .alignment import AlignedInstance
from .bootstrap import ConfidenceInterval
import numpy as np
import random
from .name_classifier import ListBasedNameGenderClassifier
from .wordlists import WordListCounter


from dataclasses import dataclass
from collections import Counter
from typing import Union, Protocol


class ParsedInstance(Protocol):
    @property
    def source(self) -> str:
        pass

    @property
    def summary(self) -> str:
        pass

    @property
    def parse(self) -> SummaryParse:
        pass
    
    @property
    def original_article_id(self) -> str:
        pass


@dataclass
class EntityInclusionStatistics:
    source_category_counts: dict[str, int]
    summary_category_counts: dict[str, int]    

def compute_entity_inclusion_stats(
    instructions: dict[str, ReplaceInstruction],
    source_to_summary_alignment: dict[str, list[int]],
) -> dict[str, int]:
    """
    Compute the entity inclusion measure for a summary and a set of instructions.
    """

    source_category_counts = Counter(i.entity_category for i in instructions.values() if i.entity_category is not None)
    summary_category_counts = Counter()

    for source_entity_id, summary_entity_ids in source_to_summary_alignment.items():
        if len(summary_entity_ids) > 0:
            summary_category_counts[instructions[source_entity_id].entity_category] += 1

    return EntityInclusionStatistics(source_category_counts, summary_category_counts)

def odds(val):
    return val / (1 - val)

def compute_adjusted_max_odds_ratio(inclusion_props: dict[str, float]) -> tuple[float, str, str]:
    pairs = [(odds(p1) / odds(p2), cat1, cat2) for cat1, p1 in inclusion_props.items() for cat2, p2 in inclusion_props.items() if cat1 != cat2]
    score, max_key, min_key = max(pairs)

    return score - 1, max_key, min_key


def compute_entity_inclusion_score_from_stat_matrix(
    source_category_counts: np.ndarray,
    summary_category_counts: np.ndarray,
) -> tuple[float, str, str]:
    probs = summary_category_counts.sum(0) / source_category_counts.sum(0)
    score, max_key, min_key = compute_adjusted_max_odds_ratio(
        {
            idx: probs[idx] for idx in range(source_category_counts.shape[-1])
        }
    )
    return score, max_key, min_key


def reshape_instance_matrices(
        instances,
        *matrices
) -> tuple[np.ndarray]:
    """
    Reshape matrices for each instance in `instances` so that they can be passed to `compute_entity_inclusion_score_from_stat_matrix`.
    """
    article_id_counts = Counter(i.original_article_id for i in instances)
    expected_sample_count = article_id_counts.most_common(1)[0][1]
    print(article_id_counts)

    for cnt in article_id_counts.values():
        if cnt != expected_sample_count:
            print(cnt, expected_sample_count)
            raise ValueError("All articles must have the same number of instances for bootstrap sampling.")
    
    return tuple(m.reshape((len(article_id_counts), expected_sample_count, m.shape[-1])) for m in matrices)


def compute_entity_inclusion_score_with_ci(
    categories: list[str],
    instances: list[AlignedInstance],
    n_bootrap_samples: int = 1000,
) -> tuple[float, ConfidenceInterval, tuple[str, str]]:
    instances = sorted(instances, key=lambda i: i.original_article_id)
    stats = [compute_entity_inclusion_stats(instance.instructions, instance.source_to_summary_alignment) for instance in instances]
    source_category_counts = np.array([[s.source_category_counts.get(cat, 0) for cat in categories] for s in stats])
    summary_category_counts = np.array([[s.summary_category_counts.get(cat, 0) for cat in categories] for s in stats])

    if n_bootrap_samples is not None:
        ci = ConfidenceInterval.from_article_samples(
            lambda *v: compute_entity_inclusion_score_from_stat_matrix(*v)[0],
            [i.original_article_id for i in instances],
            [i.sample_id for i in instances],
            source_category_counts,
            summary_category_counts,
            n_bootrap_samples=n_bootrap_samples,
        )
    else:
        ci = None
    
    score, max_key, min_key = compute_entity_inclusion_score_from_stat_matrix(source_category_counts, summary_category_counts)
    return score, ci, (categories[max_key], categories[min_key])

def compute_tvd(d1: np.array, d2: np.array) -> float:
    return np.sum(np.abs(d1 - d2)) / 2

def compute_uniform_tvd(d1: np.array) -> float:
    return compute_tvd(d1, 1 / len(d1))

def compute_hallucination_bias_from_stat_matrix(
    non_aligned_gender_counts: np.ndarray,
) -> float:
    tot = non_aligned_gender_counts.sum()
    per_cat = non_aligned_gender_counts.reshape(-1, non_aligned_gender_counts.shape[-1]).sum(axis=0) / tot

    return compute_uniform_tvd(per_cat)

def compute_entity_hallucination_score_with_ci(
    instances: list[AlignedInstance],
    name_classifier: ListBasedNameGenderClassifier,
    n_bootrap_samples: int = 1000,
) -> tuple[float, ConfidenceInterval, str]:
    instances = sorted(instances, key=lambda i: i.original_article_id)
    categories = name_classifier.get_supported_categories()

    non_aligned_gender_counts = np.zeros((len(instances), len(categories)))

    tot_cnt = Counter()
    for instance_idx, instance in enumerate(instances):
        non_aligned_summary_entity_ids = [
            entity_id for entity_id, aligned_ids in instance.summary_to_source_alignment.items() if len(aligned_ids) == 0
        ]

        hallucination_categories = Counter()
        for entity_id in non_aligned_summary_entity_ids:
            entity_tokens = list(instance.parse.named_entities[entity_id])
            if all(t.lower() in instance.source.lower() for t in entity_tokens):
                continue

            cat = name_classifier(entity_tokens)
            if cat is not None:
                hallucination_categories[cat] += 1
        
        for idx, cat in enumerate(categories):
            non_aligned_gender_counts[instance_idx, idx] = hallucination_categories[cat]
        
        tot_cnt.update(hallucination_categories)
    
        
    if n_bootrap_samples is not None:
        ci = ConfidenceInterval.from_article_samples(
            compute_hallucination_bias_from_stat_matrix,
            [i.original_article_id for i in instances],
            [i.sample_id for i in instances],
            non_aligned_gender_counts,
            n_bootrap_samples=n_bootrap_samples,
        )
    else:
        ci = None

    highest_hallucination_cat = tot_cnt.most_common(1)[0][0]

    return compute_hallucination_bias_from_stat_matrix(non_aligned_gender_counts), ci, highest_hallucination_cat


def compute_word_list_bias_from_stat_matrix(
    source_category_counts: np.ndarray,
    summary_category_counts: np.ndarray,
) -> float:
    n_cat = source_category_counts.shape[-1]
    source_probs = source_category_counts.reshape(-1, n_cat).sum(axis=0) / source_category_counts.sum()
    summary_probs = summary_category_counts.reshape(-1, n_cat).sum(axis=0) / summary_category_counts.sum()
    return compute_tvd(
        source_probs,
        summary_probs
    )

def compute_word_list_score_with_ci(
    instances: list[ParsedInstance],
    word_list_counter: WordListCounter,
    n_bootrap_samples: int = 1000,
) -> tuple[float, ConfidenceInterval]:
    """
    Compute the word list bias measure for a set of instances and word lists.
    """
    instances = sorted(instances, key=lambda i: i.original_article_id)
    categories = word_list_counter.categories

    source_category_counts = np.zeros((len(instances), len(categories)))
    summary_category_counts = np.zeros((len(instances), len(categories)))

    for instance_idx, instance in enumerate(instances):
        source_counts = word_list_counter.count_in_text(instance.source)
        #print(source_counts)
        #print([source_counts.get(cat, 0) for cat in categories])
        source_category_counts[instance_idx, :] = [source_counts.get(cat, 0) for cat in categories]
        summary_counts = word_list_counter.count_in_tokens(list(instance.parse))
        summary_category_counts[instance_idx, :] = [summary_counts.get(cat, 0) for cat in categories]

    if n_bootrap_samples is not None:
        ci = ConfidenceInterval.from_article_samples(
                compute_word_list_bias_from_stat_matrix,
                [i.original_article_id for i in instances],
                [i.sample_id for i in instances],
                source_category_counts,
                summary_category_counts,
                n_bootrap_samples=n_bootrap_samples,
        )
    else:
        ci = None

    return compute_word_list_bias_from_stat_matrix(source_category_counts, summary_category_counts), ci
