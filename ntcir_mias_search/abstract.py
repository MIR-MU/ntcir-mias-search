"""
This module defines abstract data types.
"""

from abc import abstractmethod
from logging import getLogger


LOGGER = getLogger(__name__)


class NamedEntity(object):
    """
    This class represents an entity with a unique identifier, and a human-readable text description.

    Attributes
    ----------
    identifier : str
        A unique identifier.
    description : str
        A human-readable text description.
    """

    def __str__(self):
        return "%s (look for '%s' in filenames)" % (self.description, self.identifier)

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.identifier, self.description)

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        return isinstance(other, NamedEntity) and self.identifier == other.identifier

    def __lt__(self, other):
        return isinstance(other, NamedEntity) and self.identifier < other.identifier


class Singleton(type):
    """
    This metaclass designates a class as a singleton. No more than one instance
    of the class will be instantiated.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MathFormat(NamedEntity, metaclass=Singleton):
    """
    This class represents a format in which mathematical formulae are represented.
    """
    @abstractmethod
    def encode(self, Formula):
        """
        Returns a text representation of a mathematical formula.

        Parameters
        ----------
        formula : Formula
            A mathematical formula that will be represented in this format.

        Returns
        -------
        str
            The text representation of the mathematical formula.
        """
        pass


class QueryExpansionStrategy(NamedEntity, metaclass=Singleton):
    """
    This class represents a query expansion strategy for extracting queries out of an NTCIR-10 Math,
    NTCIR-11 Math-2, and NTCIR-12 MathIR topic.
    """
    @abstractmethod
    def produce_queries(self, topic):
        """
        Produces triples of formulae, keywords, and stripe widths from a NTCIR-10 Math, NTCIR-11
        Math-2, and NTCIR-12 MathIR topic.

        Parameters
        ----------
        topic : Topic
            A topic that will serve as the source of the triples.

        Yield
        -----
        (sequence of Formula, sequence of str, int)
            Formulae, keywords, and stripe widths.
        """
        pass


class ScoreAggregationStrategy(NamedEntity):
    """
    This class represents a strategy for aggregating a real score, and a probability of relevance
    into a single aggregate score.
    """
    @abstractmethod
    def aggregate_score(self, result):
        """
        Aggregates a score assigned to a paragraph in a result by MIaS, and an estimated probability
        that the paragraph is relevant by the NTCIR Math Density Estimator package.

        Parameters
        ----------
        result : MIaSResult
            A result.

        Returns
        -------
        float
            The aggregate score of the result.
        """
        pass


class Result(object):
    """
    This class represents the result of a query.

    Attributes
    ----------
    identifier : str
        The identifier of the paragraph in the result.
    """
    @abstractmethod
    def aggregate_score(self):
        """
        Aggregates the MIaS score of the result, and the estimated probability of relevance of the
        paragraph in the result using the aggregation strategy of the query that produced this
        result.

        Returns
        -------
        float
            The aggregate score.
        """
        pass

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        return isinstance(other, Result) and self.identifier == other.identifier


class EvaluationStrategy(NamedEntity):
    """
    This class represents a strategy for evaluating result lists using relevance judgements.
    """
    @abstractmethod
    def evaluate(self, results):
        """
        Evaluates a result list.

        Parameters
        ----------
        results : ResultList
            A result list to be evaluated.

        Returns
        -------
        float
            The evaluation result.
        """
        pass
