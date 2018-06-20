"""
This module defines data types for evaluation.
"""

from logging import getLogger

from .abstract import EvaluationStrategy, Singleton
from .topic import Topic


LOGGER = getLogger(__name__)


class Bpref(EvaluationStrategy, metaclass=Singleton):
    """
    This class represents a strategy for evaluating result lists using the Bpref evaluation metric.
    (Buckley and Voorhees, 2004)
    """
    def __init__(self):
        self.identifier = "bpref"
        self.description = "Bpref"

    def evaluate(self, results):
        assert isinstance(results, ResultList)

        R = sum(1 for relevant in results.topic.judgements.values() if relevant is True)
        N = sum(1 for relevant in results.topic.judgements.values() if relevant is False)
        assert N == len(results.topic.judgements) - R
        assert min(R, N) > 0

        n = 0
        bpref = 0.0
        for result in results:
            if result.identifier in results.topic.judgements:
                if results.topic.judgements[result.identifier] is True:
                    bpref += 1.0 - (1.0 * n) / min(R, N)
                else:
                    n = min(n + 1, R)
                    assert n >= 0 and n <= min(R, N)
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
        evaluation : EvaluationStrategy, optional
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
