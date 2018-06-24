"""
This module defines data types for representing queries.
"""

from contextlib import contextmanager
from logging import getLogger
from pathlib import Path
import re

from lxml import etree
from numpy import linspace
from lxml.etree import _Element, XMLParser

from .abstract import QueryExpansionStrategy, ScoreAggregationStrategy, MathFormat, Result
from .abstract import Singleton, WeightedScoreAggregationStrategy
from .util import write_tsv
from .webmias import WebMIaSIndex


AGGREGATION_ALPHAS = linspace(0, 1, 101)
LOGGER = getLogger(__name__)
PATH_QUERY = "%s_%s.%d.query.txt"
PATH_RESPONSE = "%s_%s.%d.response.xml"
PATH_RESULT = "%s_%s.%d.results.%s.tsv"
XPATH_RESULT = ".//result"


class LeaveRightmostOut(QueryExpansionStrategy):
    """
    This class represents the Leave Rightmost Out (LRO) query expansion strategy (Růžička et al.,
    2014).
    """
    def __init__(self):
        self.identifier = "LRO"
        self.description = "Leave Rightmost Out strategy (Růžička et al. 2014)"

    def produce_queries(self, topic):
        num_queries = len(topic.formulae) + len(topic.keywords) + 1
        for query_index_keywords, first_keyword in enumerate(range(len(topic.keywords) + 1)):
            yield (
                topic.formulae, topic.keywords[first_keyword:], num_queries - query_index_keywords)
        for query_index_formulae, last_formula in enumerate(range(len(topic.formulae) - 1, -1, -1)):
            yield (
                topic.formulae[0:last_formula], topic.keywords,
                num_queries - len(topic.keywords) - query_index_formulae - 1)


class MIaSScore(ScoreAggregationStrategy, metaclass=Singleton):
    """
    This class represents a strategy for aggregating a score, and a probability estimate into
    an aggregate score. The aggregate score corresponds to the MIaS score, the probability estimate
    is discarded.
    """
    def __init__(self):
        self.identifier = "orig"
        self.description = "The original MIaS score with the probability estimate discarded"

    def aggregate_score(self, result):
        assert isinstance(result, MIaSResult)

        score = result.score
        assert isinstance(score, float)

        return score


class ArithmeticMean(WeightedScoreAggregationStrategy):
    """
    This class represents a strategy for aggregating a score, and a probability estimate into their
    weighted arithmetic mean.

    Parameters
    ----------
    alpha : float
        A value in the range [0; 1]. The weight of a score is 1 / r + (r - 1) / r * (1 - alpha),
        and the weight of a probability estimate is (r - 1) / r * alpha, where r is the rank of the
        result.
    """
    def __init__(self, alpha=0.5):
        assert isinstance(alpha, float)
        assert alpha >= 0.0 and alpha <= 1.0

        self.identifier = "arith%0.2f" % alpha
        self.description = "The weighted arithmetic mean (alpha = %0.2f)" % alpha
        self.alpha = alpha

    def aggregate_score(self, result):
        assert isinstance(result, MIaSResult)

        # rank = float(result.rank())
        # assert rank >= 1.0
        relative_rank = result.relative_rank()
        assert relative_rank >= 0.0 and relative_rank < 1.0
        score = result.score
        assert isinstance(score, float)
        assert score >= 0.0
        p_relevant = result.p_relevant
        assert isinstance(p_relevant, float)
        assert p_relevant >= 0.0 and p_relevant <= 1.0

        # w1 = 1 / rank + (rank - 1) / rank * (1 - self.alpha)
        # w2 = (rank - 1) / rank * self.alpha
        w1 = (1 - relative_rank) + relative_rank * (1 - self.alpha)
        w2 = relative_rank * self.alpha
        arithmetic_mean = score * w1 + p_relevant * w2
        return arithmetic_mean


class GeometricMean(WeightedScoreAggregationStrategy):
    """
    This class represents a strategy for aggregating a score, and a probability estimate into their
    weighted geometric mean.

    Parameters
    ----------
    alpha : float
        A value in the range [0; 1]. The weight of a score is 1 / r + (r - 1) / r * (1 - alpha),
        and the weight of a probability estimate is (r - 1) / r * alpha, where r is the rank of the
        result.
    """
    def __init__(self, alpha=0.5):
        assert isinstance(alpha, float)
        assert alpha >= 0.0 and alpha <= 1.0

        self.identifier = "geom%0.2f" % alpha
        self.description = "The weighted geometric mean (alpha = %0.2f)" % alpha
        self.alpha = alpha

    def aggregate_score(self, result):
        assert isinstance(result, MIaSResult)

        # rank = float(result.rank())
        # assert rank >= 1.0
        relative_rank = result.relative_rank()
        assert relative_rank >= 0.0 and relative_rank < 1.0
        score = result.score
        assert isinstance(score, float)
        assert score >= 0.0
        p_relevant = result.p_relevant
        assert isinstance(p_relevant, float)
        assert p_relevant >= 0.0 and p_relevant <= 1.0

        # w1 = 1 / rank + (rank - 1) / rank * (1 - self.alpha)
        # w2 = (rank - 1) / rank * self.alpha
        w1 = (1 - relative_rank) + relative_rank * (1 - self.alpha)
        w2 = relative_rank * self.alpha
        geometric_mean = score**w1 * p_relevant**w2
        return geometric_mean


class HarmonicMean(WeightedScoreAggregationStrategy):
    """
    This class represents a strategy for aggregating a score, and a probability estimate into
    their weighted harmonic mean.

    Parameters
    ----------
    alpha : float
        A value in the range [0; 1]. The weight of a score is 1 / r + (r - 1) / r * (1 - alpha),
        and the weight of a probability estimate is (r - 1) / r * alpha, where r is the rank of the
        result.
    """
    def __init__(self, alpha=0.5):
        assert isinstance(alpha, float)
        assert alpha >= 0.0 and alpha <= 1.0

        self.identifier = "harm%0.2f" % alpha
        self.description = "The weighted harmonic mean (alpha = %0.2f)" % alpha
        self.alpha = alpha

    def aggregate_score(self, result):
        assert isinstance(result, MIaSResult)

        # rank = float(result.rank())
        # assert rank >= 1.0
        relative_rank = result.relative_rank()
        assert relative_rank >= 0.0 and relative_rank < 1.0
        score = result.score
        assert isinstance(score, float)
        assert score >= 0.0
        p_relevant = result.p_relevant
        assert isinstance(p_relevant, float)
        assert p_relevant >= 0.0 and p_relevant <= 1.0

        # w1 = 1 / rank + (rank - 1) / rank * (1 - self.alpha)
        # w2 = (rank - 1) / rank * self.alpha
        w1 = (1 - relative_rank) + relative_rank * (1 - self.alpha)
        w2 = relative_rank * self.alpha
        harmonic_mean = (w1 / score + w2 / p_relevant)**-1
        return harmonic_mean


class BestScore(ScoreAggregationStrategy, metaclass=Singleton):
    """
    This class represents a strategy assigning the best possible score to a MIaS result. The score
    is either infinity, negative infinity, or zero depending on whether the result is relevant,
    non-relevant, or non-judged according to relevance judgements.
    """
    def __init__(self):
        self.identifier = "best"
        self.description = "The best possible score that uses relevance judgements"

    def aggregate_score(self, result):
        assert isinstance(result, MIaSResult)

        if result.relevant:
            score = float("inf")
        elif result.relevant is False:
            score = float("-inf")
        else:
            score = 0.0
        assert isinstance(score, float)

        return score


class WorstScore(ScoreAggregationStrategy, metaclass=Singleton):
    """
    This class represents a strategy assigning the worst possible score to a MIaS result. The score
    is either infinity, negative infinity, or zero depending on whether the result is non-relevant,
    relevant, or non-judged according to relevance judgements.
    """
    def __init__(self):
        self.identifier = "worst"
        self.description = "The worst possible score that uses relevance judgements"

    def aggregate_score(self, result):
        assert isinstance(result, MIaSResult)

        if result.relevant:
            score = float("-inf")
        elif result.relevant is False:
            score = float("inf")
        else:
            score = 0.0
        assert isinstance(score, float)

        return score


class Query(object):
    """
    This class represents a query extracted from a NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12
    MathIR topic.

    Parameters
    ----------
    topic : Topic
        The topic that will serve as the source of the query.
    math_format : MathFormat
        The format in which the mathematical formulae are represented in the query.
    payload : str
        The text content of the query.
    query_number : int
        The number of the query among all queries extracted from the topic.
    stripe_width : int
        The number of results this query will contribute to the final result list each time the
        query gets its turn.

    Attributes
    ----------
    topic : Topic
        The topic that served as the source of the query.
    payload : str
        The text content of the query.
    math_format : MathFormat
        The format in which the mathematical formulae are represented in the query.
    query_number : int
        The number of the query among all queries extracted from the topic.
    stripe_width : int
        The stripe width, i.e. number of results this query will contribute to the final result list
        each time the query gets its turn.
    response_text : str
        The text of the XML response.
    results : iterable of MIaSResult
        The query results. After the Query object has been constructed, this iterable will be empty.
        Use the finalize method to obtain the results.
    """
    def __init__(self, topic, math_format, payload, query_number, stripe_width):
        assert isinstance(math_format, MathFormat)
        assert isinstance(payload, str)
        assert isinstance(query_number, int)
        assert query_number > 0
        assert isinstance(stripe_width, int)
        assert stripe_width > 0

        self.topic = topic
        self.math_format = math_format
        self.payload = payload
        self.query_number = query_number
        self.stripe_width = stripe_width

    query_expansions = set((LeaveRightmostOut(), ))

    @staticmethod
    def from_topic(topic, math_format, query_expansion=LeaveRightmostOut()):
        """
        Produces queries from a NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12 MathIR topic.

        Parameters
        ----------
        topic : Topic
            A topic that will serve as the source of the queries.
        math_format : MathFormat
            The format in which the mathematical formulae will be represented in a query.
        query_expansion : QueryExpansionStrategy
            A query expansion strategy that produces triples of formulae, keywords, and stripe
            widths.

        Yield
        -----
        Query
            A query produced from the topic.
        """
        assert isinstance(math_format, MathFormat)
        assert isinstance(query_expansion, QueryExpansionStrategy)

        for query_number, (formulae, keywords, stripe_width) in \
                enumerate(query_expansion.produce_queries(topic)):
            for keyword in keywords:
                assert isinstance(keyword, str)
            assert stripe_width > 0

            payload_formulae = ' '.join(math_format.encode(formula) for formula in formulae)
            payload_keywords = ' '.join('"%s"' % keyword for keyword in keywords)
            payload = "%s %s" % (payload_keywords, payload_formulae)

            yield Query(topic, math_format, payload, query_number + 1, stripe_width)

    def save(self, output_directory):
        """
        Stores the text content of the query.

        Parameters
        ----------
        output_directory : Path
            The path to the directory, where the output files will be stored.
        """
        assert isinstance(output_directory, Path)

        with (output_directory / Path(PATH_QUERY % (
                self.topic.name, self.math_format.identifier, self.query_number))).open("wt") as f:
            f.write(self.payload)


class ExecutedQuery(object):
    """
    This class represents a query after it has been executed.

    Parameters
    ----------
    query : Query
        The query that has been executed.
    response_text : str
        The text of the WebMIaS response.

    Attributes
    ----------
    query : Query
        The query that has been executed.
    response_text : str
        The text of the WebMIaS response.
    """
    def __init__(self, query, response_text):
        assert isinstance(query, Query)
        assert isinstance(response_text, str)
        assert response_text

        self.query = query
        self.response_text = response_text

    @staticmethod
    def from_webmias(query, webmias):
        """
        Retrieves a query response from a deployed WebMIaS Java Servlet.

        Parameters
        ----------
        query : Query
            That query that will be executed.
        webmias : WebMIaSIndex
            The index of a deployed WebMIaS Java Servlet.

        Returns
        -------
        ExecutedQuery
            The executed query with the WebMIaS response.
        """
        assert isinstance(query, Query)
        assert isinstance(webmias, WebMIaSIndex)

        response = webmias.query(query)
        assert isinstance(response, _Element)
        response_text = etree.tostring(response, pretty_print=True).decode("utf-8")
        assert isinstance(response_text, str)

        return ExecutedQuery(query, response_text)

    def save(self, output_directory):
        """
        Stores the WebMIaS response to the query.

        Parameters
        ----------
        output_directory : Path
            The path to the directory, where the output files will be stored.
        """
        assert isinstance(output_directory, Path)

        with (output_directory / Path(PATH_RESPONSE % (
                self.query.topic.name, self.query.math_format.identifier,
                self.query.query_number))).open("wt") as f:
            f.write(self.response_text)


class ExecutedProcessedQuery(object):
    """
    This class represents a query after it has been executed, and the query results processed.

    Parameters
    ----------
    executed_query : ExecutedQuery
        The executed query with the WebMIaS response.
    positions : dict of (string, double)
        A map from paragraph identifiers to estimated positions of paragraphs in their parent
        documents. The positions are in the range [0; 1].
    estimates : sequence of double
        Estimates of P(relevant | position) in the form of a histogram.

    Attributes
    ----------
    aggregation : ScoreAggregationStrategy
        The score aggregation strategy that will be used to compute the aggregate score of the query
        results. By default, this corresponds to MIaSScore, i.e. no score aggregation will be
        performed. Change this attribute by using the use_aggregation context manager method.
    executed_query : ExecutedQuery
        The executed query whose results have been processed.
    results : sequence of MIaSResult
        The query results.
    ranks : dict of (MIaSResult, int)
        The ranks of the individual query results.
    """
    def __init__(self, executed_query, positions, estimates):
        assert isinstance(executed_query, ExecutedQuery)

        self.aggregation = MIaSScore()
        self.executed_query = executed_query

        parser = XMLParser(encoding="utf-8", recover=True)
        response = etree.fromstring(executed_query.response_text, parser=parser)
        results = [
            MIaSResult.from_element(self, result, positions, estimates)
            for result in response.xpath(XPATH_RESULT)]

        self.results = list(results)
        self.ranks = {result: rank for rank, result in enumerate(self.results)}

    @contextmanager
    def use_aggregation(self, aggregation):
        """
        Changes the score aggregation strategy, and sorts query results according to the aggregated
        scores for the duration of the context.

        Parameters
        ----------
        aggregation : ScoreAggregationStrategy
            The score aggregation strategy that will be used to compute the aggregate score of the
            query results for the duration of the context.
        """
        assert isinstance(aggregation, ScoreAggregationStrategy)

        original_aggregation = self.aggregation
        original_results = self.results
        original_ranks = self.ranks
        self.aggregation = aggregation
        if aggregation != MIaSScore():
            self.results = sorted(self.results)
            self.ranks = {result: rank for rank, result in enumerate(self.results)}
        yield
        self.aggregation = original_aggregation
        self.results = original_results
        self.ranks = original_ranks

    def save(self, output_directory):
        """
        Stores the query results as files.

        Parameters
        ----------
        output_directory : Path
            The path to the directory, where the output files will be stored.
        """
        assert isinstance(output_directory, Path)

        with (output_directory / Path(PATH_RESULT % (
                self.executed_query.query.topic.name,
                self.executed_query.query.math_format.identifier,
                self.executed_query.query.query_number,
                self.aggregation.identifier))).open("wt") as f:
            write_tsv(f, [(self.executed_query.query.topic, self.results)])


class MIaSResult(Result):
    """
    This class represents an actual result of a query to a WebMIaS Java Servlet.

    Parameters
    ----------
    query : ExecutedProcessedQuery
        The query that produced the result.
    identifier : str
        The identifier of the paragraph in the result.
    score : float
        The score of the result.
    position : float
        The estimated position of the paragraph in the original document.
    p_relevant : float
        The estimated probability of relevance of the paragraph in the result.

    Attributes
    ----------
    query : ExecutedProcessedQuery
        The query that produced the result.
    identifier : str
        The identifier of the paragraph in the result.
    score : float
        The MIaS score of the result.
    position : float
        The estimated position of the paragraph in the original document.
    p_relevant : float
        The estimated probability of relevance of the paragraph in the result.
    relevant : bool or None
        Whether the result is considered relevant according to relevance judgements.
    """
    def __init__(self, query, identifier, score, position, p_relevant, relevant):
        assert isinstance(query, ExecutedProcessedQuery)
        assert isinstance(identifier, str)
        assert isinstance(score, float)
        assert score >= 0.0
        assert isinstance(position, float)
        assert position >= 0.0 and position < 1.0
        assert isinstance(p_relevant, float)
        assert p_relevant >= 0.0 and p_relevant <= 1.0
        assert isinstance(relevant, bool) or relevant is None

        self.query = query
        self.identifier = identifier
        self.score = score
        self.position = position
        self.p_relevant = p_relevant
        self.relevant = relevant
        self._aggregate_scores = dict()

    @staticmethod
    def from_element(query, result_tree, positions, estimates):
        """
        Constructs a result from a XML element in a WebMIaS response.

        Parameters
        ----------
        query : ExecutedProcessedQuery
            The query that produced the result.
        result_tree : _Element
            A result XML element.
        positions : dict of (str, float)
            A map from paragraph identifiers to estimated positions of paragraphs in their parent
            documents. The positions are in the range [0; 1].
        estimates : sequence of float
            Estimates of P(relevant | position) in the form of a histogram.

        Returns
        -------
        MIaSResult
            The extracted result.
        """
        assert isinstance(query, ExecutedProcessedQuery)
        assert isinstance(result_tree, _Element)

        identifier_path_trees = result_tree.xpath(".//path")
        assert len(identifier_path_trees) == 1
        identifier_path_tree = identifier_path_trees[0]
        identifier_path = Path(identifier_path_tree.text)
        identifier = identifier_path.stem

        score_trees = result_tree.xpath(".//info")
        assert len(score_trees) == 1
        score_tree = score_trees[0]
        score_match = re.match(r"\s*score\s*=\s*([-0-9.]*)", score_tree.text)
        assert score_match
        score = float(score_match.group(1))
        assert score >= 0.0

        assert identifier in positions
        position = positions[identifier]
        assert isinstance(position, float)
        assert position >= 0.0 and position < 1.0

        p_relevant = estimates[int(position * len(estimates))]
        assert isinstance(p_relevant, float)
        assert p_relevant >= 0.0 and p_relevant <= 1.0

        judgements = query.executed_query.query.topic.judgements
        if identifier in judgements:
            relevant = judgements[identifier]
            assert isinstance(relevant, bool)
        else:
            relevant = None  # No relevance judgement as opposed to positive / negative judgement

        return MIaSResult(
            query, identifier, score, position, p_relevant, relevant)

    def relative_rank(self):
        """
        Returns the relative rank of this result among all results to a query.

        Returns
        -------
        float
            The relative rank of this result in the range [0; 1), where 0 is the relative rank of
            the first result.
        """
        ranks = self.query.ranks
        assert self in ranks

        rank = ranks[self]
        assert rank >= 0

        if rank == 0:
            relative_rank = 0.0
        else:
            relative_rank = 1.0 * rank / len(ranks)
        assert relative_rank >= 0.0 and relative_rank < 1.0

        return relative_rank


    def aggregate_score(self):
        """
        Aggregates the MIaS score of the result, and the estimated probability of relevance of the
        paragraph in the result using the aggregation strategy of the query that produced this
        result.

        Returns
        -------
        double
            The aggregate score.
        """
        aggregation = self.query.aggregation
        if aggregation not in self._aggregate_scores:
            aggregate_score = aggregation.aggregate_score(self)
            self._aggregate_scores[aggregation] = aggregate_score
        aggregate_score = self._aggregate_scores[aggregation]
        assert isinstance(aggregate_score, float)
        return aggregate_score

    def __lt__(self, other):
        return isinstance(other, MIaSResult) and self.aggregate_score() > other.aggregate_score()

    aggregations = set(
        [MIaSScore(), BestScore(), WorstScore()] +
        [ArithmeticMean(alpha) for alpha in AGGREGATION_ALPHAS] +
        [GeometricMean(alpha) for alpha in AGGREGATION_ALPHAS] +
        [HarmonicMean(alpha) for alpha in AGGREGATION_ALPHAS])

    def __getstate__(self):  # Do not serialize the aggregate score cache
        return (
            self.query, self.identifier, self.score, self.position, self.p_relevant,
            self.relevant)

    def __setstate__(self, state):
        self.query, self.identifier, self.score, self.position, self.p_relevant, \
            self.relevant = state
        self._aggregate_scores = dict()

    def __str__(self):
        return "%0.4f\t%0.4f\t%0.4f\t%s" % (
            self.score, self.position, self.p_relevant, self.relevant)

    def __repr__(self):
        return "%s(%s, %f, %f)" % (
            self.__class__.__name__, self.identifier, self.score, self.p_relevant)


class ArtificialResult(Result):
    """
    This class represents an artificially created result.

    Parameters
    ----------
    identifier : str
        The identifier of the paragraph in the result.
    score : float
        The artificial score of the result.

    Attributes
    ----------
    identifier : str
        The identifier of the paragraph in the result.
    score : float
        The artificial score of the result.
    """
    def __init__(self, identifier, score):
        assert isinstance(score, float)
        assert isinstance(identifier, str)

        self.score = score
        self.identifier = identifier

    def aggregate_score(self):
        return self.score

    def __getstate__(self):
        return (self.identifier, self.score)

    def __setstate__(self, state):
        self.identifier, self.score = state

    def __str__(self):
        return "%0.4f\tUNKNOWN\tUNKNOWN\t%s" % (self.score, False)
