import numpy as np
import random
from typing import Callable, Union
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ConfidenceInterval:
    article_bounds: tuple[float, float]
    sample_bounds: tuple[float, float]
    
    @classmethod
    def from_article_scores(
        cls,
        scores: list[float],
        agg_func: Callable[[list[float]], float] = np.mean,
        n_bootrap_samples: int = 1000,
    ) -> "ConfidenceInterval":
        sample_scores = []
        for _ in range(n_bootrap_samples):
            sample_scores.append(agg_func(random.choices(scores, k=len(scores))))
        
        return ConfidenceInterval(
            article_bounds=(np.percentile(sample_scores, 2.5), np.percentile(sample_scores, 97.5)),
            sample_bounds=None
        )

    @classmethod
    def from_article_samples(
        cls,
        score_func: Callable[..., float],
        article_ids: list[str],
        sample_ids: list[str],
        *values: tuple[np.ndarray, ...],
        n_bootrap_samples: int = 1000,
    ) -> "ConfidenceInterval":
        article_resampling_scores = []
        instance_resampling_scores = []

        article_sample_indices = defaultdict(lambda: defaultdict(list))

        for idx, (id_, sample_id) in enumerate(zip(article_ids, sample_ids)):
            article_sample_indices[id_][sample_id].append(idx)

        article_sample_indices_by_instance = [list(e.values()) for e in article_sample_indices.values()]
        article_sample_indices = [[v for l in e.values() for v in l] for e in article_sample_indices.values()]

        for _ in range(n_bootrap_samples):
            article_resampling_indices = [i for l in random.choices(article_sample_indices, k=len(article_sample_indices)) for i in l]
            article_matrices = [matrix[article_resampling_indices] for matrix in values]
            article_resampling_scores.append(score_func(*article_matrices))

            sample_resampling_indices = [j for l in article_sample_indices_by_instance for i in random.choices(l, k=len(l)) for j in i]
            sample_matrices = [matrix[sample_resampling_indices] for matrix in values]
            instance_resampling_scores.append(score_func(*sample_matrices))

        return ConfidenceInterval(
            article_bounds=(np.percentile(article_resampling_scores, 2.5), np.percentile(article_resampling_scores, 97.5)),
            sample_bounds=(np.percentile(instance_resampling_scores, 2.5), np.percentile(instance_resampling_scores, 97.5)),
        )

    def format_for_cli(self) -> str:
        if self.sample_bounds is None:
            return f"d: {self.article_bounds[0]:.3f}, {self.article_bounds[1]:.3f}"
        else:
            return f"d: {self.article_bounds[0]:.3f}, {self.article_bounds[1]:.3f} s: {self.sample_bounds[0]:.3f}, {self.sample_bounds[1]:.3f}"
    
    def format_for_latex(self) -> str:
        if self.sample_bounds is not None:
            return f"\\\\\\tiny s: {self.sample_bounds[0]:.2f},{self.sample_bounds[1]:.2f} \\\\\\tiny d: {self.article_bounds[0]:.2f},{self.article_bounds[1]:.2f}"
        else:
            return f"\\\\\\tiny d: {self.article_bounds[0]:.2f},{self.article_bounds[1]:.2f}"
