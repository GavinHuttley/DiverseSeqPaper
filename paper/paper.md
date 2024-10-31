---
title: '`diverse-seq`: an application for alignment-free selecting and clustering biological sequences'
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
    orcid: 0000-0002-8459-6168
    affiliation: 1
  - name: Robert McArthur
    orcid: 0000-0001-9099-339X
    affiliation: 1

affiliations:
 - name: Research School of Biology, Australian National University, Australia
   index: 1
date: 13 August 2024
bibliography: paper.bib
header-includes:
  - \input{header.tex}
  - \DeclareMathOperator*{\argmin}{argmin}
---


# Summary
<!--
summary for non specialists
--->

Bioinformatic workflows that involve computationally costly algorithms should be prototyped to reduce unnecessary computation. For instance, selecting a subset of homologous sequences can be used to identify suitable parameters for the multiple sequence alignment of tens of thousands of genomes. There are also use cases where the sequences involved are not homologous, such as can occur during machine learning projects, but representative sampling helps to avoid biases from imbalanced sequence groups.

As the size of DNA sequence datasets continues to grow, a tool that efficiently solves this problem, both statistically and computationally, is needed. `diverse-seq` implements alignment-free algorithms for identifying representatives of the diversity in a sequence collection and for clustering all the sequences.

For the first use case, we show that an entropy measure of $k$-mer frequencies allows `diverse-seq` to identify sequences that correspond well to conventional genetic distance based sampling. The computational performance for this case is linear with respect to the number of sequences and can be run in parallel. Applied to a collection of 10.5k whole microbial genomes on a laptop, `diverse-seq` took ~8 minutes to prepare the data and 4 minutes to select 100 representatives. For the second use case,

`diverse-seq` is a Python package that provides both a command-line interface and cogent3 plugins. The latter simplifies integration by users into their own analyses. It is licensed under BSD-3 and is available via the Python Package Index and GitHub.

# Statement of need

Bioinformatics data sampling workflows benefit from the selection of a subset of sequences that represent the full diversity present in large sequence collections [e.g. @parks.2018.natbiotechnol; @zhu.2019.nat.commun]. It is also the case that the compute time of algorithms such as phylogenetic estimation greatly benefit from having a good initial estimate of the phylogeny. Thus the motivation for alignment free methods is thus both computational performance and statistical accuracy.

Existing tools require input data in formats that themselves can be computationally costly to acquire. For instance, tree-based sampling procedures can be efficient, but they rely on a phylogenetic tree or a pairwise distance matrix, both of which require sequence alignment [e.g. @widmann.2006.molcellproteomics; @balaban.2019.plosone]. Thus, while tree traversal algorithms are efficient, the estimation of the can tree combine the time for sequence alignment and tree estimation.

The `diverse-seq` sequence selection algorithms are linear in time and more flexible than published approaches. It is alignment-free and does not require sequences to be related. However, in the case that the sequences are homologous, the set selected by `diverse-seq` is comparable to what would be expected under published approaches where sampling is based on genetic distance [@balaban.2019.plosone].

# Definitions

A $k$-mer is a subsequence of length $k$ and a $k$-mer probability vector has elements corresponding to the frequency of each $k$-mer in a sequence. The Shannon entropy of a probability vector is calculated as $H=-\sum_i p_i \log_2 p_i$ where $p_i$ is the probability of the $i$-th $k$-mer. As an indication of the interpretability of Shannon entropy, a DNA sequence with equifrequent nucleotides has the maximum possible $H=2$ while a sequence with a single nucleotide has $H=0$. Thus, this quantity represents a measure of "uncertainty" in the vector and is commonly used in sequence analysis, for example, to define the information content of DNA sequences as displayed in sequence logos [@schneider.1990.nucleicacidsres].

Shannon entropy is integral to other statistical measures that quantify uncertainty [@lin.1991.ieeetrans.inf.theory], including Jensen-Shannon divergence (JSD), which we employ in this work. As illustrated in Table 1, the magnitude of JSD reflects the level of relatedness amongst sequences via the similarity between their $k$-mer probability vectors.

For a collection of DNA sequences $\mathbb{S}$ with size $N$, define $f_i$ as the $k$-mer frequency vector for sequence $s_i, s_i \in \mathbb{S}$. The JSD for the resulting set of vectors, $\mathbb{F}$, is

\begin{equation*}
JSD(\mathbb{F})=H \left( \frac{1}{N}\sum_i^N f_i \right) - \overline{H(\mathbb{F})}
\end{equation*}

where the first term corresponds to the Shannon entropy of the mean of the $N$ probability vectors and the second term $\overline{H(\mathbb{F})}$ is the mean of their corresponding Shannon entropies. For vector $f_i$, $f_i \in \mathbb{F}$, its contribution to the total JSD of $\mathbb{F}$ is

\begin{equation}
\delta_{JSD}(i)=JSD(\mathbb{F})-JSD(\mathbb{F} - \{f_i\})
\end{equation}\label{eqn:delta-jsd}

From the equation, it is apparent that to update the JSD of a collection efficiently, we need only track $k$-mer counts, total Shannon entropy and the number of sequences in the collection. Thus, the algorithm can be implemented with a single pass through the data.

To facilitate the description below, we define the record with the minimum $\delta_{JSD}$ as $$lowest = \argmin_{i \in N} \delta_{JSD}(i)$$

# Algorithms

## Selection of representative sequences 

The algorithm for computing the Jensen-Shannon divergence is quite simple. What follows are the optimisations we have employed to make the calculations scalable in terms of the number of sequences.

1. Sequence data is BLOSC2 compressed as unsigned-8 bit integers and saved in HDF5 format on disk.
2. `numba`, a just-in-time compiler, is used for the core algorithms producing $k$-mers and their counts, providing a significant speed up over a pure python implementation [@numba].
3. Sequence loading and $k$-mer counting is triggered when a sequence record is considered for inclusion in the divergent set, reducing the memory required to that for the user-nominated size.

The `prep` sub-command converts plain text sequence data into an on-disk storage format that is more efficient for access in the other steps. A user can provide either fasta or GenBank formatted DNA sequence files. The sequences are converted into unsigned 8-bit integer `numpy` arrays and stored in a single HDF5 file on disk. The resulting `.dvseqs` file is required for the `max` and `nmost` sub-commands.

The `nmost` algorithm defines an exact number of sequences to be selected that maximise the JSD. The order of input sequences is randomised and the selected set is initialised with the first $n$ sequences. As shown in \autoref{algo:nmost}, for each of the remaining sequences, if adding it to the set $\mathbb{F} - {lowest}$ increases JSD, it replaces $lowest$. The `max` algorithm differs from `nmost` by defining lower and upper bounds for the number of sequences in the divergent set. It further amends the within-loop condition (\autoref{algo:max}), allowing the number of sequences in the set to change when a statistical measure of $\delta_{JSD}$ variance increases. We provide users a choice of two measures of variance in $\delta_{JSD}$: the standard deviation or the coefficient of variation.

## Constructing trees from $k$-mers

The mash distance [@ondov.2016.mash] is an unbiased estimate of the mutation rate between two sequences that can
be computed in near linear time for large genome size. It estimates the mutation rate from an approximation of the
Jaccard index between the $k$-mer sets of the two sequences (the percentage of shared $k$-mers). The approximation
is calculated from only a random subset of all $k$-mers in the two sequences. This subset for a sequence is called the MinHash sketch. 
The size of the MinHash sketch is called the *sketch size* and contains the smallest $k$-mers when sorted by a hash function. We use agglomerative clustering with average linkage [@murtagh.2012.algorithms] based on pairwise mash distances to estimate phylogenetic trees from unaligned sequences. The resulting tree depends on the $k$-mer size and the sketch size.

## `dvs` command line application

- `prep` converts sequences into numpy arrays for faster processing
- `nmost` samples the $n$ sequences that increase JSD most.
- `max` samples sequences that maximise a user specified statistic, either the standard deviation or the coefficient of variation of $\delta_{JSD}$.
- `ctree` quick estimations of phylogenetic trees from unaligned sequences using mash distances.

## `dvs` cogent3 apps

We provide `dvs_nmost` and `dvs_max` as cogent3 apps. For users with cogent3 installed, `dvs_nmost` and `dvs_max` are available at runtime via the cogent3 function `get_app()`. The apps mirror the settings from their command-line implementation but differ in that they operate directly on a sequence collection, skipping conversion to disk storage and directly returning the selected subset of sequences. This is demonstrated in the `plugin_demo.ipynb` notebook.

# Performance

## Selection of representative sequences
### Recovery of representatives from synthetic knowns

We evaluate the ability of `dvs max` to recover known divergent lineages using simulated data. The design was intended to assess whether rare sequences can be recovered. We defined 4 distinct sequence compositions and two distinct "pool" compositions: *balanced*, in which each sequence family was present at equal frequency, or *imbalanced*, where one sequence occurred at 1%, another 49% and the remainder at 25% each. In each scenario, we simulated a total of 50 sequences. If `dvs max`  identifies a set of precisely 4 sequences with one pool representative this is counted as a success. As shown in \autoref{fig:synthetic-knowns}, the primary determinant of the success was the length of the simulated sequences.

### The selected sequences are phylogenetically diverse

For homologous DNA sequences, increasing the amount of elapsed time since they shared a common ancestor increases their genetic distance due to time-dependent accumulation of  sequence changes. We expect that the JSD between two sequences will also increase proportional to the amount of time since they last shared a common ancestor. We therefore pose the null hypothesis that if JSD is not informative, then the minimum pairwise genetic distance amongst $N$ sequences chosen by `diverse_seq` will be approximately equal to the minimum pairwise genetic distance between a random selection of $N$ sequences. Under the alternate hypothesis that JSD is informative, the minimum genetic distance between sequences chosen by `diverse_seq` will be larger than between randomly selected sequences. We test this hypothesis using a resampling statistic [@sokal.1995, 808], estimating the probability of the algorithmic choice being consistent with the null hypothesis. This probability is calculated as the proportion of 1000 randomly selected sets of sequences whose minimum genetic distance was greater or equal to that obtained from the sequences chosen by `dvs max`. We further summarised the performance of the `dvs` commands as the percentage of loci which gave a $p$-value less than 0.05. A bigger percentage is better.

We addressed the above hypothesis using 106 alignments of protein coding DNA sequences from the following 31 mammals: Alpaca, Armadillo, Bushbaby, Cat, Chimp, Cow, Dog, Dolphin, Elephant, Gorilla, Hedgehog, Horse, Human, Hyrax, Macaque, Marmoset, Megabat, Microbat, Mouse, Orangutan, Pig, Pika, Platypus, Rabbit, Rat, Shrew, Sloth, Squirrel, Tarsier, Tenrec and Wallaby. The sequences were obtained from Ensembl.org [@harrison.2024.nucleicacidsresearch] and aligned using cogent3's codon aligner [@knight.2007.genomebiol]. The genetic distance between the sequences was calculated using the paralinear distance [@lake.1994.procnatlacadsciua].

The results of the analysis (\autoref{fig:jsd-v-dist}) indicated the sucess of `dvs max` in identifying genetically diverse sequences was principally sensitive to the choice of $k$. While \autoref{fig:jsd-v-dist}(a) showed close equivalence between the statistics, \autoref{fig:jsd-v-dist}(b) indicates the size of the selected set using the standard deviation was systematically lower than for the coefficient of variation. The result from the `dvs nmost` analysis, which performed using the minimum set size argument given to `dvs max` is represented by the $JSD(\mathbb{F})$ statistic.

### Computational performance

As shown in \autoref{fig:compute-time}, the compute time was linear with respect to the number of sequences, shown using random samples of microbial genomes from the 960 REFSOIL dataset [@choi.2017.ismej]. We further trialled the algorithm on the dataset of @zhu.2019.nat.commun, which consists of 10,560 whole microbial genomes.  Using 10 cores on a MacBook Pro M2 Max, application of `dvs prep` followed by `dvs nmost` took 8'9" and 3'45" (to select 100 sequences) respectively. The RAM memory per process was ~300MB.

## Constructing trees from $k$-mers

We use the mammals dataset to evaluate the statistical performance of the method. All sequences were concatenated and phylogenetic trees were estimated from this alignment with different $k$-mer sizes and sketch sizes. The trees generated by `dvs ctree` are compared to the maximum likelihood tree found by IQ-TREE [@minh.2020.iq] using a general time-reversible model [@tavare.1986.some] on the concatenated alignment.

\autoref{fig:ctree-k} shows how the likelihood of the generated trees changes as $k$ increases for varying sketch sizes. When $k \leq 5$, the $k$-mers are non-unique (all mash-distances are zero) and the method generates a caterpillar tree. For larger $k$-mer sizes the trees are more interesting, though it is not until $k=8$ when the caterpillar tree is statistically outperformed. As $k$ increases further, the likelihood approaches that of IQ-TREE but plateaus before it at $k=12$. \autoref{fig:ctree-ss} shows how the likelihood of the generated trees changes as the sketch size increases for varying $k \geq 8$. The likelihood trends upwards as the sketch size increases (that is the mash distance estimate improves), approaching but not reaching the optimal maximum likelihood found by IQ-TREE. For optimal values of $k$, the performance of the method plateaus beyond a sketch size of about 2,500. The optimal maximum likelihood achieved by `dvs ctree` is impressive considering that the method does not require an alignment.

### Computational performance

While the time complexity of the standard algorithm for agglomerative clustering is $\mathcal{O}\left(n^3\right)$, with respect to the number of sequences, the number of sequeunces is often small in comparison to the sequence lengths. As such, it was found that the most time consuming step of the algorithm was within the distance calculation. The pairwise distance calculation is done in two steps. First, is the construction of what is called the MinHash sketch for each sequence (essentially a random subset of $k$-mers of size *sketch size*), followed by the computation of distances from these sketches [@ondov.2016.mash]. The expected runtime for constructing the MinHash sketch for a sequence is $\mathcal{O}\left(l + s\log s \log l\right)$ where $l$ is the length of the sequence, and $s$ is the *sketch size*. This is essentially linear with respect to $l$ when $l$ is much larger than $s$. Hence, the time complexity for constructing all MinHash sketches is linear with respect to the combined length of all sequences. The time complexity for calculating the distance between two sequences from the MinHash sketch is $\mathcal{O}\left(s\right)$. Hence, the time complexity for calculating the pairwise distance matrix between all pairs of sequences from the sketches is $\mathcal{O}\left(sn^2\right)$. For suitable applications of this algorithm however, both the *sketch size* and the number of sequences are numerically dominated by the combined length of the sequences. Thus, the expected time to run the algorithm is linear with respect to the combined length of the sequences. This has been verified empirically with the 960 REFSOIL dataset (\autoref{fig:ctree-time}). The figure shows that as the number of sequences grows (and hence the combined length of the sequences with it), the time taken to construct the cluster tree grows linearly.

# Recommendations

For large-scale analyses, we recommend use of the `nmost` command line tool. The choice of $k$ should be guided by the maximum number of unique $k$-mers in a DNA sequence of length $L$, indicated as the result of the expression $log(L/4)$. For instance, $k\approx 12$ for bacterial genomes (which are of the order $10^6$bp). For individual protein coding genes, as \autoref{fig:jsd-v-dist} indicates, $k=6$ for `nmost` gives a reasonable accuracy. 

# TODO's

- [ ] upload the big datasets to Zenodo

# Figures

![The `diverse-seq nmost` algorithm.](figs/nmost.pdf){#algo:nmost}

![The `diverse-seq max` algorithm. This includes upper and lower bounds for the size of the divergent set and amends the within-loop condition of `nmost`. The set size is increased when a record that increases JSD also increases the standard deviation of $\delta_{JSD}$.](figs/max.pdf){#algo:max}


![Identification of representatives of known groups is affected by sequence length. `dvs max` identified representatives of known groups in both *balanced*, and *imbalanced* pools. (TODO: check correctness of simulation labels, why is 1k seqs more variable for balanced stdev?)](figs/synthetic_known_bar.pdf){#fig:synthetic-knowns}

![The statistical performance of `dvs max` in recovering representative sequences is a function of $k$ and the chosen statistic. The minimum and maximum allowed set sizes were 5 and 30, respectively. `dvs nmost` is represented by $JSD(\mathbb{F})$ run with n=5. Trendlines were estimated using LOWESS [@cleveland.1979.j.am.stat.assoc]. (a) *Significant %* is the percentage of cases where `dvs max` was significantly better ($p-\text{value} \le 0.05$) at selecting divergent sequences than a random selection process. (b) The mean and standard deviations of the number of sequences selected by `dvs max`.](figs/jsd_v_dist.pdf){#fig:jsd-v-dist}

![Result of applying the `dvs_max` app to a single sequence alignment. The phylogenetic tree was estimated using Neighbour-Joining [@saitou.1987.mol.biol.evol] from the pairwise paralinear distances [@lake.1994.procnatlacadsciua]. See the `plugin_demo` notebook for the code used to produce this figure.](figs/selected_edges.pdf){#fig:selected-edges}

![`dvs max` exhibits linear time performance with respect to the number of microbial genome sequences. Three replicates were performed for each condition. For each repeat, sequences were randomly sampled without replacement from the 960 REFSOIL microbial dataset [@choi.2017.ismej].](figs/compute_time.pdf){#fig:compute-time}

![Statistical performance of the `dvs_ctree` app on the concatenated mammals alignment as the $k$-mer size increases. The likelihood of trees generated by the app is compared to the maximum likelihood tree found by IQ-TREE [@minh.2020.iq]. For large enough sketch sizes, the likelihood approaches that of IQ-TREE and plateaus beyond a $k$-mer size of ~12.](figs/likelihood_vs_k_for_ss.pdf){#fig:ctree-k}

![Statistical performance of the `dvs_ctree` app on the concatenated mammals alignment as the sketch size increases. The likelihood of trees generated by the app is compared to the maximum likelihood tree found by IQ-TREE2 [@minh.2020.iq]. For optimal $k$-mer sizes, the likelihood approaches that of IQ-TREE and plateaus beyond a sketch size of ~2500.](figs/likelihood_vs_ss_for_k.pdf){#fig:ctree-ss}

![Computational performance of the `dvs_ctree` app on 960 REFSOIL microbial dataset. The time taken to run the algorithm grows linearly with respect to the number of sequences.](figs/ctree_times/ss_2500/time_vs_numseqs.png.pdf){#fig:ctree-time}

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