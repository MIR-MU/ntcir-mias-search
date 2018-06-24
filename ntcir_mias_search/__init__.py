"""
The NTCIR MIaS Search package implements the Math Information Retrieval system
that won the NTCIR-11 Math-2 main task (Růžička et al., 2014).
"""

from .abstract import NamedEntity, Singleton, MathFormat, Result, EvaluationStrategy  # noqa:F401
from .abstract import QueryExpansionStrategy, ScoreAggregationStrategy  # noqa:F401
from .abstract import WeightedScoreAggregationStrategy  # noqa:F401
from .eval import ResultList, Bpref  # noqa:F401
from .facade import get_topics, get_webmias, query_webmias, rerank_and_merge_results  # noqa:F401
from .query import LeaveRightmostOut  # noqa:F401
from .query import MIaSScore, ArithmeticMean, GeometricMean, HarmonicMean  # noqa:F401
from .query import MIaSResult, BestScore, WorstScore, ArtificialResult  # noqa:F401
from .query import Query, ExecutedQuery, ExecutedProcessedQuery  # noqa:F401
from .topic import Topic, Formula, TeX, PMath, CMath, PCMath  # noqa:F401
from .util import remove_namespaces, write_tsv, log_sequence  # noqa:F401
from .util import get_judgements, get_positions, get_estimates  # noqa:F401
from .view import plot_evaluation_results  # noqa:F401
from .webmias import WebMIaSIndex  # noqa:F401


__author__ = "Vit Novotny"
__version__ = "0.2.2"
__license__ = "MIT"
