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
from Random_Walk.HeterogeneousSG import HeterogeneousSG
from Random_Walk.utils import read_graph, set_seed
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

def multi_metapath_walk(
    G,
    node_types: Dict[str, str],
    start_node: str,
    metapaths: List[List[str]],
    walk_length: int,
    max_retry_per_step: int = 50
) -> List[str]:
    walk = [start_node]
    current_node = start_node
    current_type = node_types.get(current_node)

    while len(walk) < walk_length:
        # 找出当前类型能起始的路径
        candidate_paths = [mp for mp in metapaths if mp[0] == current_type]
        if not candidate_paths:
            break  # 无法继续游走

        retry_count = 0
        success = False

        while retry_count < max_retry_per_step:
            metapath = random.choice(candidate_paths)  # 随机选择一个路径
            node = current_node
            path_nodes = [node]
            metapath_success = True

            for next_type in metapath[1:]:
                neighbors = list(G.neighbors(node))
                valid_neighbors = [n for n in neighbors if node_types.get(n) == next_type]
                if not valid_neighbors:
                    metapath_success = False
                    break
                node = random.choice(valid_neighbors)
                path_nodes.append(node)

            if metapath_success:
                walk.extend(path_nodes[1:])  # 去掉重复起点
                current_node = path_nodes[-1]
                current_type = node_types.get(current_node)
                success = True
                break  # 成功走一段路径，退出 retry

            retry_count += 1

        if not success:
            break  # 多次尝试都失败，终止本次游走

    return walk

def _generate_single_walk_worker(args, G, node_types, metapaths, walk_length):
    """处理单个游走任务的worker函数"""
    node, task_seed = args
    random.seed(task_seed)
    walk = multi_metapath_walk(
        G=G,
        node_types=node_types,
        start_node=node,
        metapaths=metapaths,
        walk_length=walk_length,
        max_retry_per_step=50
    )
    return walk

def generate_metapath_walks(
    G: nx.Graph,
    node_types: Dict[str, str],
    metapaths: List[List[str]],
    walk_length: int,
    num_walks: int,
    seed: int = 43,
    workers: int = os.cpu_count()
) -> List[List[str]]:
    random.seed(seed)
    walks = []
    # all_nodes = list(G.nodes())

    # 只选择类型为 drug 或 disease 的节点作为起点
    start_nodes = [n for n in G.nodes() if node_types.get(n) in ['drug', 'disease']]

    # for node in start_nodes:
    #     for _ in range(num_walks):
    #         walk = multi_metapath_walk_with_fallback(G, node_types, node, metapaths, walk_length)
    #         if len(walk) > 1:
    #             walks.append(walk)
    # 准备任务参数列表 (node, task_seed)
    tasks = []
    for node_idx, node in enumerate(start_nodes):
        for walk_idx in range(num_walks):
            # 为每个游走生成独立但确定的种子
            task_seed = seed + node_idx * num_walks + walk_idx
            tasks.append((node, task_seed))

    # 创建partial函数固定共享参数
    worker_func = partial(
        _generate_single_walk_worker,
        G=G,
        node_types=node_types,
        metapaths=metapaths,
        walk_length=walk_length
    )

    # 并行执行游走生成
    with multiprocessing.Pool(processes=workers) as pool:
        results = pool.map(worker_func, tasks)

    # 过滤空游走
    walks = [walk for walk in results if len(walk) > 1]
    return walks

def save_embedding_files(netf:str,  outputf:str, nodetypef:str=None, seed:int=43,
                         directed:bool=False, weighted:bool=True,
                         num_walks:int=100, walk_length:int=5, workers:int=os.cpu_count(),
                         dimension:int=128, window_size:int=4, p:float=1, q:float=1, 
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
        walk_length=walk_length,
        num_walks=num_walks,
        seed=seed
    )

    with open('results/M-RW_results/tmp_walk_file.pkl','wb') as fw:
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

