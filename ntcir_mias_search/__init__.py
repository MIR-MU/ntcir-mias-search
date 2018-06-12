"""
The NTCIR MIaS Search package implements the Math Information Retrieval system
that won the NTCIR-11 Math-2 main task (Růžička et al., 2014).
"""

from .processing import get_topics, get_webmias, query_webmias, rerank_and_merge_results  # noqa:F401


__author__ = "Vit Novotny"
__version__ = "0.1.0"
__license__ = "MIT"
