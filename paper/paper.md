---
title: '`diverse-seq`: a linear-time application for selecting representative biological sequences'
tags:
  - Python
  - genomics
  - statistics
  - machine learning
  - bioinformatics
  - molecular evolution
  - phylogenetics
authors:
  - name: Gavin Huttley
    orcid: 0000-0001-7224-2074
    affiliation: 1
  - name: Katherine Caley
    affiliation: 1
  - name: Robert McArthur??
    affiliation: 1

affiliations:
 - name: Research School of Biology, Australian National University, Australia
   index: 1
date: 13 August 2024
bibliography: paper.bib
header-includes:
  - \input{header.tex}
---

<!--

see this paper as an example https://joss.theoj.org/papers/10.21105/joss.06875
 and the instructions on their markdown format https://joss.readthedocs.io/en/latest/paper.html
-->



# Summary
<!--
summary for non specialists
--->

There are many bioinformatic workflows that can benefit by the selection of a representative subset of much larger datasets. A subset that efficiently captures the diversity in a large sample can, for instance, avoid sample imbalance in selecting data for machine learning projects. Such a subset of homologous sequences, for instance, can also be efficiently applied to parameter tuning for multiple sequence alignment of tens-of-thousands of genomes. Existing solutions to the problem of selecting a representative subset of sequences may rely on pre-processing steps that are computationally inefficient and / or not applicable to the full range of potential applications. For instance, a phylogenetic tree provides a good basis for sampling representative sequences but the computational cost of producing that tree can be prohibitive.

As the size of DNA sequence datasets continues to grow, there is need for a tool that efficiently solves this problem, both statistically and computationally. To address this need, we have developed `diverse-seq`, an alignment-free algorithm that identifies representatives of the diversity in a sequences collection. We show that the entropy measure of $k$-mer frequencies employed by `diverse-seq` allow it to identify sequences that correspond well with genetic distance based sampling. The computational performance of `diverse-seq` is linear with respect to the number of sequences and can be run in parallel. Applied to a collection of 10.5k whole microbial genomes on a high-end latop, `diverse-seq` took ~8 minutes to prepare the data and 4 minutes to select 100 representatives.

`diverse-seq` is a Python package that provides both a command-line interface and cogent3 plugins. The latter simplifies integration by users into their own analyses. It is licensed under BSD-3 and available on the Python Package Index and GitHub.

# Statement of need

Bioinformatics data sampling workflows benefit by being able to select a subset of sequences that represent the full diversity present in a large sequence collections [e.g. @parks.2018.natbiotechnol; @zhu.2019.nat.commun]. For some analyses, including groups of highly related sequences imposes a significant computational cost for no information gain (cite). In some circumstances, retention of such related groups can lead to biases in estimation (cite). The motivation for selecting representative groups of sequences can thus be both computational performance and statistical accuracy.

Other tools require the existence of input data formats that themselves can be computationally costly to acquire. For instance, tree-based sampling procedures can be efficient but rely on a phylogenetic tree or a pairwise distance matrix, both of which require sequence alignment [e.g. @widmann.2006.molcellproteomics; @balaban.2019.plosone] (todo: check what source Balaban method presumes their tree comes from). Thus, while tree traversal algorithms are efficient, the estimation of the tree may not be. The same holds for distance estimation.

The `diverse-seq` algorithm is linear in time and more flexible than published approaches. It is alignment-free and does not require sequences to be related. However, if the sequences are homologous, the set selected by `diverse-seq` is comparable to what would be expected under published approaches.

# Definitions

A $k$-mer is a subsequence of length $k$ and a $k$-mer probability vector has elements corresponding to the frequency of each $k$-mer in a sequence. The Shannon entropy of a probability vector is calculated as $H=-\sum_i p_i \log_2 p_i$ where $p_i$ is the probability of the $i$-th $k$-mer. As an indication of the interpretability of Shannon entropy, a DNA sequence with equifrequent nucleotides has the maximum possible $H=2$ while a sequence with a single nucleotide has $H=0$. Thus, this quantity represents a measure of "uncertainty" in the vector and is commonly used in sequence analysis, for example, to define the information content of DNA sequences as displayed in sequence logos [@schneider.1990.nucleicacidsres].

Shannon entropy is integral to other statistical measures that quantify uncertainty [@lin.1991.ieeetrans.inf.theory], including Jensen-Shannon divergence (JSD), which we employ in this work. As illustrated in Table 1, the magnitude of JSD reflects the level of relatedness amongst sequences via the similarity between their $k$-mer probability vectors.

For a collection of DNA sequences $\mathbb{S}$ with size $N$, define $f_i$ as the $k$-mer frequency vector for sequence $s_i, s_i \in \mathbb{S}$. The JSD for the resulting set of vectors, $\mathbb{F}$, is

\begin{equation*}
JSD(\mathbb{F})=H \left( \frac{1}{N}\sum_i^N f_i \right) - \overline{H(\mathbb{F})}
\end{equation*}

where the first term corresponds to the Shannon entropy of the mean of the $N$ probability vectors and the second term $\overline{H(\mathbb{F})}$ is the mean of their corresponding Shannon entropies. For vector $f_i$, $f_i \in \mathbb{F}$, its contribution to the total JSD of $\mathbb{F}$ is

\begin{equation}
\delta_{JSD}(i)=JSD(\mathbb{F})-JSD(\mathbb{F} - \{i\})
\end{equation}\label{eqn:delta-jsd}

From the equation, it is apparent that to update the JSD of a collection efficiently, we need only track $k$-mer counts, total Shannon entropy and the number of sequences in the collection. Thus, the algorithm can be implemented with a single pass through the data.

# Algorithm

The algorithm for computing the Jensen-Shannon divergence is quite simple. What follows are the optimisations we have employed to make the calculations scalable in terms of the number of sequences.

1. Sequence data is saved BLOSC2 compressed as unsigned-8 bit integers in HDF5 storage on disk.
2. A $k$-mer is identified as an index in a $4^k$ vector with counts stored with sufficient integer precision to capture the vector's maximum element.
3. $k$-mers are only counted when a sequence record is considered for inclusion in the divergent set, reducing the memory required to that for the user-nominated size.
4. We use `numba` for just-in-time compilation of core algorithms for producing $k$-mers and their counts.

The `dvs prep` sub-command converts plain text sequence data into an on disk storage format that is more efficient for access in the other steps. A user can provide either fasta or GenBank formatted DNA sequence files. The sequences are converted into unsigned 8-bit integer `numpy` arrays and stored in a single HDF5 file on disk. The resulting `.dvseqs` file is required for the `max` and `nmost` commands.

> **Info**
> Possibly better served with a figure than embedding linted code?

**Algorithm 1** The `diverse-seq` `nmost` algorithm.\label{algorithm:nmost}

```python
records: list[KmerSeq]  # a list of sequence converted into k-mer counts
min_size: int  # the minimum size of the divergent set
max_size: int  # the maximum size of the divergent set
shuffle(records)  # randomise the order of the records

# SummedRecords sorts records by their delta-JSD. The record
# with the lowest delta-JSD is excluded from the N-1 set.
sr = SummedRecords.from_records(records[:min_size])
for r in records:
    if sr.increases_jsd(r):
      sr = sr.replaced_lowest(r)
```

If the input sequence collection consisted of equally distant sequences (in terms of a genetic distance measure), then a representative set would exhibit maximum variance in $\delta_{JSD}$. We provide users a choice of two measures of variance in  $\delta_{JSD}$  in the `dvs max` algorithm: the standard deviation or the coefficient of variation (Algorithm 2).

**Algorithm 2** The `diverse-seq`  `max` algorithm. This amends the within-loop condition of `nmost` to the following, where `std` represents the standard deviation of $\delta_{JSD}$.\label{algorithm:max}

```python
if sr.increases_jsd(r):
  # adding r to the N-1 set increased JSD over sr.jsd
  nsr = sr + r  # create a new set with the current set and r
  sr = nsr if nsr.std > sr.std else sr.replaced_lowest(r)
  # if the new set has a higher standard deviation, keep the new set
  if sr.size > max_size:
    # stay within the user specified limits by dropping the lowest
    sr = sr.dropped_lowest()
```

## `dvs` command line application

- `prep` converts sequences into numpy arrays for faster processing
- `nmost` samples the $n$ sequences that increase JSD most.
- `max` samples diverse sequences that maximise a user specified statistic, either the standard deviation or the coefficient of variation of $\delta_{JSD}$.

## `dvs` cogent3 plugins

We provide `dvs_nmost` and `dvs_max` as Cogent3 apps, made available to users at runtime via cogent3 `get_app()`. The apps mirror the settings from their command-line application but differ in that they operate directly on a sequence collection (skipping conversion to disk storage), returning the selected subset of sequences. This is demonstrated in the `plugin_demo.ipynb` notebook.

# Performance

## Recovery of representatives from synthetic knowns

We evaluate the ability of `dvs max` to recover known divergent lineages. We defined 4 distinct sequence compositions and two distinct "pool" compositions: *balanced*, in which each sequence family was present at equal frequency, or *imbalanced*, where one sequence occurred at 1%, another 49% and the remainder at 25% each. In each scenario, we simulated a total of 50 sequences. This design was intended to assess whether rare sequences can be recovered. If `dvs max`  identifies a set of precisely 4 sequences with one pool representative this is counted as a success. As shown in \autoref{fig:synthetic-knowns}, the primary determinant of the success was the length of the simulated sequences.

## The selected sequences are phylogenetically diverse

For homologous DNA sequences, increasing the amount of elapsed time since they shared a common ancestor increases their genetic distance due to time-dependent accumulation of  sequence changes. We expected that the JSD between two sequences will also increase proportional to the amount of time since they last shared a common ancestor. From these we pose the null hypothesis that if JSD is uninformative, then the minimum pairwise genetic distance amongst $N$ sequences chosen by `diverse_seq` will be approximately equal to the minimum pairwise genetic distance between a random selection of $N$ sequences. Under the alternate hypothesis, the minimum genetic distance between sequences chosen by `diverse_seq` will be larger than between randomly selected sequences. We test this hypothesis using a resampling statistic [@sokal.1995] (TODO: Gavin find page), estimating the probability of the algorithmic choice being consistent with the null hypothesis as the proportion of 1000 randomly selected sets of sequences whose minimum genetic distance was greater or equal to that obtained from the sequences chosen by `dvs max`. We further summarised the performance of the `dvs` commands as the percentage of loci which gave a $p$-value less than 0.05. A bigger percentage is better.

We addressed the above question using 106 alignments of protein coding DNA sequences from the following 31 mammals: Alpaca, Armadillo, Bushbaby, Cat, Chimp, Cow, Dog, Dolphin, Elephant, Gorilla, Hedgehog, Horse, Human, Hyrax, Macaque, Marmoset, Megabat, Microbat, Mouse, Orangutan, Pig, Pika, Platypus, Rabbit, Rat, Shrew, Sloth, Squirrel, Tarsier, Tenrec and Wallaby. The sequences were obtained from Ensembl.org [@harrison.2024.nucleicacidsresearch] and aligned using cogent3's codon aligner [@knight.2007.genomebiol]. The genetic distance between the sequences was calculated using the paralinear distance [lake.1994.procnatlacadsciua].

The results of the analysis (\autoref{fig:jsd-v-dist}) indicated the sucess of `dvs max` in identifying genetically diverse sequences was principally sensitive to the choice of $k$. While \autoref{fig:jsd-v-dist}(a) showed close equivalence between the statistics, \autoref{fig:jsd-v-dist}(b) indicates the size of the selected set using the standard deviation was systematically lower than for the coefficient of variation. The result from the `dvs nmost` analysis, which (performed using the minimum set size argument given to `dvs max`) is represented by the $JSD(\mathbb{F})$ statistic.

## Computational performance

As shown in \autoref{fig:compute-time}, the compute time was linear with respect to the number of sequences on random samples of the microbial genomes. We further trialled the algorithm on the data set of @zhu.2019.nat.commun, which consists of 10560 whole microbial genomes.  Using 10 cores on a MacBook Pro M2 Max, application of `dvs prep` followed by `dvs nmost` took 8'9" and 3'45" (to select 100 sequences) respectively. The RAM memory per process was ~300MB.

# Recommendations

For large-scale analyses, we recommend the command line `nmost` tool. The choice of $k$ should be guided by the maximum number of unique $k$-mers in a DNA sequence of length $L$, indicated as the result of the expression $log(L/4)$. For instance, $k\approx 12$ for bacterial genomes (which are of the order $10^6$bp). For that case, as \autoref{fig:jsd-v-dist} indicates, $k=6$ for `nmost` gives a reasonable approximation.

# TODO's

- [ ] read the Balaban paper
- [ ] check the Sokal reference
- [ ] upload the big data sets to Zenodo

# Figures

![Identification of representatives of known groups is affected by sequence length. `dvs max` identified representatives of known groups in both *balanced*, and *imbalanced* pools. (TODO: check correctness of simulation labels, why is 1k seqs more variable for balanced stdev?)](figs/synthetic_known_bar.png){#fig:synthetic-knowns}

![The statistical performance of `dvs max` in recovering representative sequences is a function of $k$ and the chosen statistic. The minimum and maximum allowed set sizes were 5 and 30, respectively. `dvs nmost` is represented by $JSD(\mathbb{F})$ run with n=5. Trendlines were estimated using LOWESS [@cleveland.1979.j.am.stat.assoc]. (a) *Significant%* is the percentage of cases where `dvs max` was significantly better ($p-\text{value} \le 0.05$) at selecting divergent sequences than a random selection process. (b) The mean and standard deviations of the number of sequences selected by `dvs max`.](figs/jsd_v_dist.png){#fig:jsd-v-dist}

![Result of applying the `dvs_max` app to a single sequence alignment. The phylogenetic tree was estimated using Neighbour-Joining [@saitou.1987.mol.biol.evol] from the pairwise paralinear distances [@lake.1994.procnatlacadsciua]. See the `plugin_demo` notebook for the code used to produce this figure.](figs/selected_edges.png){#fig:selected-edges}

![`dvs max` exhibits linear time performance with respect to the number of microbial genome sequences. Three replicates were performed for each condition. For each repeat, sequences were randomly sampled without replacement from the 960 REFSOIL microbial data set [@choi.2017.ismej].](figs/compute_time.png){#fig:compute-time}

# Tables


**Table 1** Jensen-Shanon Divergence (JSD) for different relationships between two sequences.\label{JSD-examples}

+--------------+--------+--------+-----+
| Relationship |   seq1 |   seq2 | JSD |
+==============+========+========+=====+
|    Identical | `ATCG` | `TCGA` | 0.0 |
+--------------+--------+--------+-----+
|   No overlap | `AAAA` | `TTTT` | 1.0 |
+--------------+--------+--------+-----+
| Intermediate | `ATCG` | `ATCC` | 0.5 |
+==============+========+========+=====+


# Acknowledgements

blah blah

# References