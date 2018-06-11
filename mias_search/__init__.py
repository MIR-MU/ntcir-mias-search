"""
The MIaS Search package implements the Math Information Retrieval system that
won the NTCIR-11 Math-2 main task (Růžička et al., 2014).
"""

from .processing import get_results, merge_results, rerank_results, Topic, WebMIaSIndex


__author__ = "Vit Novotny"
__version__ = "0.1.0"
__license__ = "MIT"
