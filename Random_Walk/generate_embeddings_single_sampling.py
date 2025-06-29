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

def match_metapath(walk: List[str], node_types: Dict[str, str], metapaths: List[List[str]]) -> bool:
    walk_types = [node_types.get(n) for n in walk]
    return any(walk_types == mp for mp in metapaths)

def guided_random_walk_until_terminal(
    G,
    node_types: Dict[str, str],
    start_node: str,
    metapaths: List[List[str]],
    max_steps: int = 100
) -> List[str]:
    start_type = node_types.get(start_node)

    for _ in range(max_steps):  # 尝试最多 max_steps 次游走
        walk = [start_node]
        current_node = start_node

        for _ in range(max_steps):  # 每次游走最多走 max_steps 步
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            walk.append(next_node)
            next_type = node_types.get(next_node)

            if next_type in ['drug', 'disease']:
                if next_type == start_type:
                    break  # 同类型，回退重新开始
                else:
                    if match_metapath(walk, node_types, metapaths):
                        return walk  # 合法终点，符合路径
                    else:
                        break  # 不符合路径，回退重新开始

            current_node = next_node

    return []  # 所有尝试都失败，返回空路径




def _generate_guided_walk_worker(args, G, node_types, metapaths):
    node, seed, num_walks = args
    random.seed(seed)
    walks = []

    for _ in range(num_walks):
        walk = guided_random_walk_until_terminal(
            G=G,
            node_types=node_types,
            start_node=node,
            metapaths=metapaths
        )
        if walk:
            walks.append(walk)
            # print(f"Try walk: {walk} | Types: {[node_types[n] for n in walk]}")

    return walks



def generate_metapath_walks(
    G: nx.Graph,
    node_types: Dict[str, str],
    metapaths: List[List[str]],
    num_walks: int,
    seed: int = 43,
    workers: int = os.cpu_count()
) -> List[List[str]]:
    start_nodes = [n for n in G.nodes() if node_types.get(n) in ['drug', 'disease']]
    tasks = [(n, seed + idx * num_walks, num_walks) for idx, n in enumerate(start_nodes)]

    worker_func = partial(
        _generate_guided_walk_worker,
        G=G,
        node_types=node_types,
        metapaths=metapaths
    )

    with multiprocessing.Pool(processes=workers) as pool:
        results = pool.map(worker_func, tasks)

    walks = [walk for group in results for walk in group]
    return walks

# def generate_terminal_guided_walks_no_parallel(
#     G: nx.Graph,
#     node_types: Dict[str, str],
#     metapaths: List[List[str]],
#     num_walks: int,
#     seed: int = 42
# ) -> List[List[str]]:
#     random.seed(seed)
#     start_nodes = [n for n in G.nodes() if node_types.get(n) in ['drug', 'disease']]
#     walks = []
#
#     for node in start_nodes:
#         for _ in range(num_walks):
#             walk = guided_random_walk_until_terminal(
#                 G=G,
#                 node_types=node_types,
#                 start_node=node,
#                 metapaths=metapaths
#             )
#             if walk:
#                 walks.append(walk)
#
#     return walks


def save_embedding_files(netf:str,  outputf:str, nodetypef:str=None, seed:int=43,
                         directed:bool=False, weighted:bool=True,
                         num_walks:int=100,
                         dimension:int=128, window_size:int=4,
                         net_delimiter:str='\t',):

    set_seed(seed)

    print('Reading network files...')
    G=read_graph(netf,weighted=weighted,directed=directed,
                 delimiter=net_delimiter)

    node_types = read_node_types(nodetypef)
    METAPATHS = [
        ['drug', 'gene', 'disease'],
        ['disease', 'gene', 'drug'],
        ['drug', 'gene', 'gene', 'disease'],
        ['disease', 'gene', 'gene', 'drug'],
        ['drug', 'gene', 'gene','gene', 'disease'],
        ['disease', 'gene', 'gene', 'gene', 'drug'],
        ['drug', 'gene', 'gene', 'gene', 'gene','disease'],
        ['disease', 'gene', 'gene', 'gene', 'gene','drug'],
        # ['drug', 'gene', 'gene', 'gene', 'gene', 'gene', 'disease'],
        # ['disease', 'gene', 'gene','gene', 'gene', 'gene','drug']
    ]

    print('Generating metapath-guided walks...')
    walks = generate_metapath_walks(
        G,
        node_types,
        metapaths=METAPATHS,
        num_walks=num_walks,
        seed=seed
    )

    with open('results/MetaPath_strict/tmp_walk_file.pkl','wb') as fw:
        pickle.dump(walks,fw)
    # with open('results/tmp_walk_file.pkl','rb') as fw:
    #     walks=pickle.load(fw)
    print('Generating node embeddings...')
    use_hetSG = True if nodetypef != None else False
    embeddings = HeterogeneousSG(use_hetSG, walks, set(G.nodes()), nodetypef=nodetypef,
                                 embedding_size=dimension, window_length=window_size, workers=1)
    with open(outputf,'wb') as fw:
        pickle.dump(embeddings,fw)

    print(f'Node embeddings saved: {outputf}')

