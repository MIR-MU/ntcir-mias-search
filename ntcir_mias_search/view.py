"""
This module contains plotting functions.
"""

from .abstract import WeightedScoreAggregationStrategy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa:E402


FIGURE_HEIGHT = 4
LINE_WIDTH = 2
SUBPLOT_WIDTH = 8


def plot_evaluation_results(math_formats):
    """
    Plots the results of an evaluation for weighted score aggregation strategies.

    Parameters
    ----------
    math_formats : iterable of (MathFormat, iterable of (WeightedScoreAggregationStrategy, float))
        Math formats associated with score aggregation strategies, and the corresponding evaluation
        results.

    Returns
    -------
    matplotlib.figure.Figure
        The plotted figure.
    """
    fig = plt.figure(figsize=(SUBPLOT_WIDTH * len(math_formats), FIGURE_HEIGHT))
    for i, (math_format, aggregations) in enumerate(math_formats):
        ax = fig.add_subplot(1, len(math_formats), i + 1)
        ax.title.set_text(math_format.identifier)
        lines = dict()
        for (aggregation, score) in aggregations:
            assert isinstance(aggregation, WeightedScoreAggregationStrategy)
            assert isinstance(score, float)

            if aggregation.__class__ not in lines:
                lines[aggregation.__class__] = ([], [])
            lines[aggregation.__class__][0].append(aggregation.alpha)
            lines[aggregation.__class__][1].append(score)
        for aggregation_class, (unsorted_x, unsorted_y) in lines.items():
            x, y = zip(*sorted(zip(unsorted_x, unsorted_y)))
            ax.plot(x, y, label=aggregation_class.__name__, linewidth=LINE_WIDTH)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$\alpha$")
        ax.grid()
        ax.legend()
    return fig
