NTCIR MIaS Search – Our search engine for the NTCIR Math tasks
==============================================================
[![CircleCI](https://circleci.com/gh/MIR-MU/ntcir-mias-search/tree/master.svg?style=shield)][ci]

 [ci]: https://circleci.com/gh/MIR-MU/ntcir-mias-search/tree/master (CircleCI)

NTCIR MIaS Search is a Python 3 command-line utility that operates on top of
[WebMIaS][] and that implements the Math Information Retrival system that won
the NTCIR-11 Math-2 main task (see the [task paper][aizawaetal14-ntcir11], and
the [system description paper][ruzickaetal14-math]).

Experimentally, NTCIR MIaS Search also reranks subquery results according to
the relevance probability estimates from the [NTCIR Math Density
Estimator][ntcir-math-density] package.

[aizawaetal14-ntcir11]: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.686.444&rep=rep1&type=pdf (NTCIR-11 Math-2 Task Overview)
[ntcir-math-density]: https://github.com/MIR-MU/ntcir-math-density (NTCIR Math Density Estimator)
[ruzickaetal14-math]: http://research.nii.ac.jp/ntcir/workshop/OnlineProceedings11/pdf/NTCIR/Math-2/07-NTCIR11-MATH-RuzickaM.pdf (Math Indexer and Searcher under the Hood: History and Development of a Winning Strategy)
[WebMIaS]: https://github.com/MIR-MU/WebMIaS (WebMIaS)

Usage
=====
Installing
----------
The package can be installed by executing the following command:

    $ pip install ntcir-mias-search

Displaying the usage
--------------------
Usage information for the package can be displayed by executing the following
command:

    $ ntcir-mias-search --help
    usage: ntcir-mias-search [-h] --dataset DATASET --topics TOPICS --positions
                             POSITIONS --estimates ESTIMATES --webmias-url
                             WEBMIAS_URL
                             [--webmias-index-number WEBMIAS_INDEX_NUMBER]
                             [--num-workers-querying NUM_WORKERS_QUERYING]
                             [--num-workers-merging NUM_WORKERS_MERGING]
                             --output-directory OUTPUT_DIRECTORY

    Use topics in the NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12 MathIR format
    to query the WebMIaS interface of the MIaS Math Information Retrieval system
    and to retrieve result document lists.

    optional arguments:
      -h, --help            show this help message and exit
      --dataset DATASET     A path to a directory containing a dataset in the
                            NTCIR-11 Math-2, and NTCIR-12 MathIR XHTML5 format.
                            The directory does not need to exist, since the path
                            is only required for extracting data from the file
                            with estimated positions of paragraph identifiers.
      --topics TOPICS       A path to a file containing topics in the NTCIR-10
                            Math, NTCIR-11 Math-2, and NTCIR-12 MathIR format.
      --positions POSITIONS 
                            The path to the file, where the estimated positions of
                            all paragraph identifiers from our dataset were stored
                            by the NTCIR Math Density Estimator package.
      --estimates ESTIMATES 
                            The path to the file, where the density, and
                            probability estimates for our dataset were stored by
                            the NTCIR Math Density Estimator package.
      --webmias-url WEBMIAS_URL
                            The URL at which a WebMIaS Java Servlet has been
                            deployed.
      --webmias-index-number WEBMIAS_INDEX_NUMBER
                            The numeric identifier of the WebMIaS index that
                            corresponds to the dataset. Defaults to 0.
      --num-workers-querying NUM_WORKERS_QUERYING
                            The number of processes that will send queries to
                            WebMIaS. Defaults to 1. Note that querying, reranking,
                            and merging takes place simmultaneously.
      --num-workers-merging NUM_WORKERS_MERGING
                            The number of processes that will rerank results.
                            Defaults to 3. Note that querying, reranking, and
                            merging takes place simmultaneously.
      --output-directory OUTPUT_DIRECTORY
                            The path to the directory, where the output files will
                            be stored.
      --plots PLOTS [PLOTS ...]
                            The path to the files, where the evaluation results
                            will plotted.

Querying WebMIaS
----------------
The following command queries a local WebMIaS instance using 64 worker
processes:

    $ mkdir search_results
    
    $ ntcir-mias-search --num-workers-querying 8 --num-workers-merging 56 \
    >     --dataset ntcir-11-12 \
    >     --topics NTCIR11-Math2-queries-participants.xml \
    >     --judgements NTCIR11_Math-qrels.dat \
    >     --estimates estimates.pkl.gz --positions positions.pkl.gz \
    >     --webmias-url http://localhost:58080/WebMIaS --webmias-index-number 1 \
    >     --plots plot.pdf plot.svg \
    >     --output-directory search_results
    Reading relevance judgements from NTCIR11_Math-qrels.dat
    50 judged topics and 2500 total judgements in NTCIR11_Math-qrels.dat
    Reading topics from NTCIR11-Math2-queries-participants.xml
    50 topics (NTCIR11-Math-1, NTCIR11-Math-2, ...) contain 55 formulae, and 113 keywords
    Establishing connection with a WebMIaS Java Servlet at http://localhost:58080/WebMIaS
    Reading paragraph position estimates from positions.pkl.gz
    8301578 total paragraph identifiers in positions.pkl.gz
    Reading density, and probability estimates from estimates.pkl.gz
    Querying WebMIaSIndex(http://localhost:58080/WebMIaS, 1), reranking and merging results
    Using 3 strategies to aggregate MIaS scores with probability estimates:
    - The best possible score that uses relevance judgements (look for 'best' in filenames)
    - The original MIaS score with the probability estimate discarded (look for 'orig' in filenames)
    - The worst possible score that uses relevance judgements (look for 'worst' in filenames)
    Storing reranked per-query result lists in search_results
    Using 4 formats to represent mathematical formulae in queries:
    - Content MathML XML language (look for 'CMath' in filenames)
    - Combined Presentation and Content MathML XML language (look for 'PCMath' in filenames)
    - Presentation MathML XML language (look for 'PMath' in filenames)
    - The TeX language by professor Knuth (look for 'TeX' in filenames)
    Result list for topic NTCIR11-Math-9 contains only 188 / 1000 results, sampling the dataset
    Result list for topic NTCIR11-Math-17 contains only 716 / 1000 results, sampling the dataset
    Result list for topic NTCIR11-Math-26 contains only 518 / 1000 results, sampling the dataset
    Result list for topic NTCIR11-Math-39 contains only 419 / 1000 results, sampling the dataset
    Result list for topic NTCIR11-Math-43 contains only 924 / 1000 results, sampling the dataset
    get_results:  100%|███████████████████████████████████████████████| 50/50 [00:26<00:00,  1.88it/s]
    rerank_and_merge_results: 200it [01:02,  3.18it/s]
    Storing final result lists in mias_search_results
    100%|█████████████████████████████████████████████████████████████| 12/12 [00:13<00:00,  3.73it/s]
    Evaluation results:
    - best, PCMath: 0.5569
    - best, PMath: 0.5283
    - best, TeX: 0.5076
    - best, CMath: 0.4983
    - orig, PCMath: 0.4917
    - ...
    - orig, PMath: 0.4616
    - worst, CMath: 0.3080
    - worst, TeX: 0.2810
    - worst, PMath: 0.1156
    - worst, PCMath: 0.1141
    Plotting plot.svg
    Plotting plot.pdf
    
    $ ls search_results
    final_CMath.best.tsv
    final_CMath.orig.tsv
    final_CMath.worst.tsv
    final_PCMath.best.tsv
    final_PCMath.orig.tsv
    final_PCMath.worst.tsv
    final_PMath.best.tsv
    final_PMath.orig.tsv
    final_PMath.worst.tsv
    final_TeX.best.tsv
    final_TeX.orig.tsv
    final_TeX.worst.tsv
    NTCIR11-Math-10_CMath.1.query.txt
    NTCIR11-Math-10_CMath.1.response.xml
    NTCIR11-Math-10_CMath.1.results.best.tsv
    NTCIR11-Math-10_CMath.1.results.orig.tsv
    NTCIR11-Math-10_CMath.1.results.worst.tsv
    NTCIR11-Math-10_CMath.2.query.txt
    NTCIR11-Math-10_CMath.2.response.xml
    ...

The following command queries a [remote WebMIaS instance][WebMIaS-demo] using
64 worker processes:

    $ mkdir search_results
    
    $ ntcir-mias-search --num-workers-querying 8 --num-workers-merging 56 \
    >     --dataset ntcir-11-12 \
    >     --topics NTCIR11-Math2-queries-participants.xml \
    >     --judgements NTCIR11_Math-qrels.dat \
    >     --estimates estimates.pkl.gz --positions positions.pkl.gz \
    >     --webmias-url https://mir.fi.muni.cz/webmias-demo --webmias-index-number 0 \
    >     --plots plot.pdf plot.svg \
    >     --output-directory search_results
    Reading relevance judgements from NTCIR11_Math-qrels.dat
    50 judged topics and 2500 total judgements in NTCIR11_Math-qrels.dat
    Reading topics from NTCIR11-Math2-queries-participants.xml
    50 topics (NTCIR11-Math-1, NTCIR11-Math-2, ...) contain 55 formulae, and 113 keywords
    Establishing connection with a WebMIaS Java Servlet at https://mir.fi.muni.cz/webmias-demo
    Reading paragraph position estimates from positions.pkl.gz
    8301578 total paragraph identifiers in positions.pkl.gz
    Reading density, and probability estimates from estimates.pkl.gz
    Querying WebMIaSIndex(https://mir.fi.muni.cz/webmias-demo, 0), reranking and merging results
    Using 3 strategies to aggregate MIaS scores with probability estimates:
    - The best possible score that uses relevance judgements (look for 'best' in filenames)
    - The original MIaS score with the probability estimate discarded (look for 'orig' in filenames)
    - The worst possible score that uses relevance judgements (look for 'worst' in filenames)
    Storing reranked per-query result lists in search_results
    Using 4 formats to represent mathematical formulae in queries:
    - Content MathML XML language (look for 'CMath' in filenames)
    - Combined Presentation and Content MathML XML language (look for 'PCMath' in filenames)
    - Presentation MathML XML language (look for 'PMath' in filenames)
    - The TeX language by professor Knuth (look for 'TeX' in filenames)
    get_results:  100%|███████████████████████████████████████████████| 50/50 [05:29<00:00,  6.58s/it]
    rerank_and_merge_results: 200it [06:57,  2.09s/it]
    Storing final result lists in mias_search_results
    100%|█████████████████████████████████████████████████████████████| 12/12 [00:13<00:00,  3.73it/s]
    Evaluation results:
    - best, PCMath: 0.5569
    - best, PMath: 0.5283
    - best, TeX: 0.5076
    - best, CMath: 0.4983
    - orig, PCMath: 0.4917
    - ...
    - orig, PMath: 0.4616
    - worst, CMath: 0.3080
    - worst, TeX: 0.2810
    - worst, PMath: 0.1156
    - worst, PCMath: 0.1141
    Plotting plot.svg
    Plotting plot.pdf
    
    $ ls search_results
    final_CMath.best.tsv
    final_CMath.orig.tsv
    final_CMath.worst.tsv
    final_PCMath.best.tsv
    final_PCMath.orig.tsv
    final_PCMath.worst.tsv
    final_PMath.best.tsv
    final_PMath.orig.tsv
    final_PMath.worst.tsv
    final_TeX.best.tsv
    final_TeX.orig.tsv
    final_TeX.worst.tsv
    NTCIR11-Math-10_CMath.1.query.txt
    NTCIR11-Math-10_CMath.1.response.xml
    NTCIR11-Math-10_CMath.1.results.best.tsv
    NTCIR11-Math-10_CMath.1.results.orig.tsv
    NTCIR11-Math-10_CMath.1.results.worst.tsv
    NTCIR11-Math-10_CMath.2.query.txt
    NTCIR11-Math-10_CMath.2.response.xml
    ...

[WebMIaS-demo]: https://mir.fi.muni.cz/webmias-demo/ (Web Math Indexer and Searcher)

Contributing
============

To get familiar with the codebase, please consult the UML class diagram in the
[Umbrello][www:Umbrello] project document [project.xmi](project.xmi):

![Rendered UML class diagram](project.svg)

[www:Umbrello]: https://umbrello.kde.org/ (Umbrello Project - Welcome to Umbrello - The UML Modeller)

Citing NTCIR MIaS Search
========================
Text
----
RŮŽIČKA, Michal, Petr SOJKA and Martin LÍŠKA. Math Indexer and Searcher under
the Hood: History and Development of a Winning Strategy. In Noriko Kando, Hideo
Joho, Kazuaki Kishida. *Proceedings of the 11th NTCIR Conference on Evaluation
of Information Access Technologies.* Tokyo: National Institute of Informatics,
2-1-2 Hitotsubashi, Chiyoda-ku, Tokyo 101-8430 Japan, 2014. p. 127-134, 8 pp.
ISBN 978-4-86049-065-2.

BibTeX
------
``` bib
@inproceedings{mir:MIaSNTCIR-11,
     author = "Michal R\r{u}\v{z}i\v{c}ka and Petr Sojka and Michal L{\' i}\v{s}ka",
      title = "{Math Indexer and Searcher under the Hood:
               History and Development of a Winning Strategy}",
      month = Dec,
       year = 2014,
    address = "Tokyo",
  booktitle = "{Proc. of the 11th NTCIR Conference on Evaluation
               of Information Access Technologies}",
     editor = "Hideo Joho and Kazuaki Kishida",
  publisher = "{NII, Tokyo, Japan}",
      pages = "127--134",
}
```
