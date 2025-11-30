"""
Evaluation metrics and utilities.
"""

from .lm_metrics import evaluate_perplexity
from .synthetic_metrics import compute_accuracy_with_details, print_examples_summary

__all__ = [
    "evaluate_perplexity",
    "compute_accuracy_with_details",
    "print_examples_summary",
]

