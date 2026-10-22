#!/usr/bin/env python3
"""
build_networks.py -- genotype networks of the binary single- and multi-peak models
==================================================================================

For every abundance snapshot <root>/Abundances/<stem>.txt (one individual per line: the
comma-separated mutated loci, or '--' for the ancestral genotype) this writes

    <root>/Networks/<stem>.graphml

with one node per distinct genotype (node id = the genotype exactly as written in the abundance
file, node attribute `abundance` = number of copies) and an edge between every pair of genotypes
at Hamming distance one. For binary genotypes two such genotypes differ by one locus, so all edges
are found by deleting one mutated locus from every genotype and looking the result up (no
all-pairs comparison).

The topology scripts of single_peak/ and multi_peak/ read these files (and the abundance
snapshots) and extract the largest connected component themselves.

    python build_networks.py Data/single_peak --times 100
    python build_networks.py Data/multi_peak --times 39 79 119 159 199 --match delta_5 --n-cores 8
"""
import argparse
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import networkx as nx


def label(genotype):
    return ",".join(map(str, genotype)) if genotype else "--"


def read_counts(path):
    counts = Counter()
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                counts[() if s == "--" else tuple(sorted(int(x) for x in s.split(",")))] += 1
    return counts


def genotype_graph(counts):
    G = nx.Graph()
    for g, n in counts.items():
        G.add_node(label(g), abundance=int(n))
    for g in counts:
        for i in range(len(g)):
            h = g[:i] + g[i + 1:]
            if h in counts:
                G.add_edge(label(g), label(h))
    return G


def build(path):
    out = path.parent.parent / "Networks" / (path.stem + ".graphml")
    G = genotype_graph(read_counts(path))
    tmp = out.with_suffix(".graphml.part")
    nx.write_graphml(G, tmp)
    tmp.replace(out)
    return f"{out.name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", help="directory containing Abundances/")
    ap.add_argument("--times", type=int, nargs="*", default=None, help="only these snapshots (default: all)")
    ap.add_argument("--match", default=None, help="only file names containing this substring")
    ap.add_argument("--n-cores", type=int, default=1)
    a = ap.parse_args()
    files = []
    for p in sorted(Path(a.root, "Abundances").glob("dataOut_*.txt")):
        m = re.search(r"_t_(\d+)$", p.stem)
        if m and (a.times is None or int(m.group(1)) in a.times) and (a.match is None or a.match in p.name):
            files.append(p)
    Path(a.root, "Networks").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.n_cores) as ex:
        for msg in ex.map(build, files):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
