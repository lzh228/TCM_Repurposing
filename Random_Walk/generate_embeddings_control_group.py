import argparse
import os
import math
import random
import pickle
import networkx as nx
import numpy as np   
from scipy import stats
import parmap
from collections import defaultdict, Counter
import multiprocessing
from DREAMwalk.HeterogeneousSG import HeterogeneousSG
from DREAMwalk.utils import read_graph, set_seed
from typing import List, Dict
from functools import partial

def read_node_types(nodetype_file: str) -> Dict[str, str]:
    """
    Reads node type from file. Format: node_id <tab> type
    """
    node_types = {}
    with open(nodetype_file, 'r') as f:
        for line in f:
            node, ntype = line.strip().split()
            node_types[node] = ntype
    return node_types

def _generate_random_walk_single(args, G):
    start_node, walk_length, seed = args
    random.seed(seed)
    walk = [start_node]
    current = start_node
    for _ in range(walk_length - 1):
        neighbors = list(G.neighbors(current))
        if neighbors:
            current = random.choice(neighbors)
            walk.append(current)
        else:
            break
    return walk if len(walk) > 1 else None

def generate_random_walks_control_parallel(
    G: nx.Graph,
    walk_length: int,
    num_walks: int,
    num_start_nodes: int,
    seed: int = 43,
    workers: int = os.cpu_count()
) -> List[List[str]]:
    random.seed(seed)
    all_nodes = list(G.nodes())
    start_nodes = random.sample(all_nodes, num_start_nodes)

    # 准备任务参数列表 (start_node, walk_length, seed)
    tasks = []
    for idx, node in enumerate(start_nodes):
        for j in range(num_walks):
            walk_seed = seed + idx * num_walks + j
            tasks.append((node, walk_length, walk_seed))

    worker_func = partial(_generate_random_walk_single, G=G)

    with multiprocessing.Pool(processes=workers) as pool:
        results = pool.map(worker_func, tasks)

    # 去除None值
    walks = [r for r in results if r is not None]
    return walks

def run_random_walk_control_experiments(
    netf: str,
    output_dir: str,
    nodetypef: str = None,
    num_trials: int = 10,
    num_walks: int = 100,
    walk_length: int = 100,
    dimension: int = 128,
    window_size: int = 4,
    directed: bool = False,
    weighted: bool = True,
    net_delimiter: str = '\t'
):
    G = read_graph(netf, weighted=weighted, directed=directed, delimiter=net_delimiter)
    node_types = read_node_types(nodetypef)
    start_node_count = len([n for n in G.nodes() if node_types.get(n) in ['drug', 'disease']])
    all_nodes_set = set(G.nodes())

    for i in range(num_trials):
        seed = 43 + i  # 不同随机种子
        print(f'Generating random walks for trial {i} with seed {seed}...')
        walks = generate_random_walks_control_parallel(
            G,
            walk_length=walk_length,
            num_walks=num_walks,
            num_start_nodes=start_node_count,
            seed=seed
        )

        with open(f'results/control_embeddings/tmp_walk_file{i}.pkl', 'wb') as fw:
            pickle.dump(walks, fw)

        print('Generating embeddings...')
        use_hetSG = True if nodetypef is not None else False
        embeddings = HeterogeneousSG(use_hetSG, walks, all_nodes_set, nodetypef=nodetypef,
                                     embedding_size=dimension, window_length=window_size, workers=1)

        outfile = os.path.join(output_dir, f'control_randomwalk_embedding_trial{i}.pkl')
        with open(outfile, 'wb') as fw:
            pickle.dump(embeddings, fw)
        print(f'Embeddings saved to {outfile}')

