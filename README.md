# Introduction
NTCIR MIaS Search is a Python 3 command-line utility that implements the Math
Information Retrival system that won the NTCIR-11 Math-2 main task (see the
[task paper][paper:aizawaetal14-ntcir11], and the [system description
paper][paper:ruzickaetal14-math]):

1. NTCIR MIaS Search loads [topics][www:ntcir-task-data] in the [NTCIR-10
   Math][paper:aizawaetal13-ntcir10], [NTCIR-11
   Math-2][paper:aizawaetal14-ntcir11], and [NTCIR-12
   MathIR][paper:zanibbi16-ntcir12] format.

2. NTCIR MIaS Search expands the topics into subqueries using the Leave
   Rightmost Out (LRO) query expansion strategy and submits the subqueries to
   [WebMIaS][www:WebMIaS].

3. NTCIR MiaS Search reranks the subquery results according using relevance
   probability estimates from the [NTCIR Math Density
   Estimator][www:ntcir-math-density] package, and produces a final result list
   by interleaving the subquery result lists. The final result list is stored
   in the TSV (Tab Separated Value) format, which is meant to be passed to the
   [MIREval][www:MIREval] tool.

[paper:aizawaetal14-ntcir11]: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.686.444&rep=rep1&type=pdf (NTCIR-11 Math-2 Task Overview)
[paper:ruzickaetal14-math]: http://research.nii.ac.jp/ntcir/workshop/OnlineProceedings11/pdf/NTCIR/Math-2/07-NTCIR11-MATH-RuzickaM.pdf (Math Indexer and Searcher under the Hood: History and Development of a Winning Strategy)

[www:MIaS]: https://github.com/MIR-MU/MIaS (MIaS)
[www:MIREval]: https://github.com/MIR-MU/MIREval (MIREval)
[www:ntcir-task-data]: https://www.nii.ac.jp/dsc/idr/en/ntcir/ntcir-taskdata.html (Downloading NTCIR Test Collections Task Data)
[www:ntcir-math-density]: https://github.com/MIR-MU/ntcir-math-density (NTCIR Math Density Estimator)
[www:WebMIaS]: https://github.com/MIR-MU/WebMIaS (WebMIaS)

# Usage
Installing:

    $ pip install ntcir-mias-search

Displaying the usage:

    $ ntcir-mias-search

<!-- TODO -->
