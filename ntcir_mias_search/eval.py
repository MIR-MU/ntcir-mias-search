"""
This module defines data types for evaluation.
"""

from logging import getLogger

from .abstract import EvaluationStrategy, Singleton
from .topic import Topic


LOGGER = getLogger(__name__)


class Bpref(EvaluationStrategy, metaclass=Singleton):
    """
    This class represents a strategy for aggregating a score, and a probability estimate into
    an aggregate score. The aggregate score corresponds to the MIaS score, the probability estimate
    is discarded.
    """
    def __init__(self):
        self.identifier = "bpref"
        self.description = "Bpref"

    def evaluate(results):
        assert isinstance(results, ResultList)

        R = sum(results.topic.judgements.values())
        N = len(results.topic.judgements.values()) - R
        n = 0
        bpref = 0.0
        for result in results:
            if result.identifier not in results.topic.judgements:
                n = min(n + 1, R)
            else:
                bpref += 1.0 - n / min(R, N)
        bpref /= R

        return bpref


class ResultList(object):
    """
    This class represents a final list of results together with the original topic.

    Parameters
    ----------
    topic : Topic
        The original topic that produced the results.
    results : sequence of Result
        The final list of results.

    Attributes
    ----------
    topic : Topic
        The original topic that produced the results.
    results : sequence of Result
        The final list of results.
    """
    def __init__(self, topic, results):
        assert isinstance(topic, Topic)

        self.topic = topic
        self.results = results
        self._evaluation_results = dict()

    def __iter__(self):
        for result in self.results:
            yield result

    def evaluate(self, evaluation=Bpref()):
        """
        Evaluates the result list.

        Parameters
        ----------
        evaluation : EvaluationStrategy
            The strategy for evaluation the result list.

        Returns
        -------
        The result of the evaluation.
        """
        assert isinstance(evaluation, EvaluationStrategy)

        if evaluation not in self._evaluation_results:
            self._evaluation_results[evaluation] = evaluation.evaluate(self)
        return self._evaluation_results[evaluation]

    def __getstate__(self):  # Do not serialize the evaluation result cache
        return (self.topic, self.results)

    def __setstate__(self, state):
        self.topic, self.results = state
        self._evaluation_results = dict()

    def __eq__(self, other):
        return isinstance(other, ResultList) and self.evaluate() == other.evaluate()

    def __lt__(self, other):
        return isinstance(other, ResultList) and self.evaluate() > other.evaluate()
