import csv
import os.path
import itertools
import time
from joblib import Parallel, delayed
from pygosemsim import download, graph, annotation, similarity, term_set
import pandas as pd


def parse_drug_target(file_path):
    drug_targets = {}

    # 打开并读取CSV文件
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            drug = row['tcm_id']
            gene = row['Gene Symbol']

            # 如果药物已经在字典中，添加基因到对应的列表中
            if drug in drug_targets:
                drug_targets[drug].append(gene)
            else:
                # 如果药物不在字典中，创建新的列表
                drug_targets[drug] = [gene]
    print(f'\n> Done parsing drug targets: read {len(drug_targets)} total drugs')

    return drug_targets


def parse_disease_genes(file_path):
    disease_genes = {}

    # 打开并读取CSV文件
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            disease = row['Symptom']
            gene = row['Symbol']

            # 如果疾病已经在字典中，添加基因到对应的列表中
            if disease in disease_genes:
                disease_genes[disease].append(gene)
            else:
                # 如果疾病不在字典中，创建新的列表
                disease_genes[disease] = [gene]
    print(f'\n> Done parsing disease genes: read {len(disease_genes)} total diseases')

    return disease_genes


def get_go_terms(proteins, protein_annot):
    """获取蛋白质的GO术语集合（带注释缓存）"""
    terms = set()
    for pid in proteins:
        if pid in protein_annot:
            terms.update(protein_annot[pid]["annotation"].keys())
    return list(terms)


def get_all_ancestors(go_term, go_graph, visited=None):
    """
    递归获取go_term的所有祖先GO术语。
    假设go_graph提供了get_parents(term)方法，返回该GO术语的直接父节点列表。
    """
    if visited is None:
        visited = set()
    try:
        parents = go_graph.get_parents(go_term)
    except AttributeError:
        # 如果go_graph不是这种结构，则尝试将其当作字典使用
        # print(dir(go_graph))
        parents = list(go_graph.predecessors(go_term))  # 使用 predecessors() 获取父节点

    ancestors = set()
    for p in parents:
        if p not in visited:
            visited.add(p)
            ancestors.add(p)
            ancestors.update(get_all_ancestors(p, go_graph, visited))
    return ancestors


def expand_go_terms(go_terms, go_graph):
    """
    对输入的GO术语集合进行扩展，包含每个GO术语的所有祖先。
    """
    expanded = set(go_terms)
    for term in go_terms:
        ancestors = get_all_ancestors(term, go_graph)
        expanded.update(ancestors)
    return expanded


def build_drug_term_matrix(drug_go_mapping):
    """
    构建药物-GO术语二进制矩阵。
    每行代表一个药物，每列代表一个GO术语，若该药物包含该术语则标记为1，否则为0。
    """
    all_terms = set()
    for terms in drug_go_mapping.values():
        all_terms.update(terms)
    all_terms = sorted(all_terms)
    matrix = pd.DataFrame(0, index=drug_go_mapping.keys(), columns=all_terms, dtype=int)
    for drug, terms in drug_go_mapping.items():
        for term in terms:
            matrix.loc[drug, term] = 1
    return matrix


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    union = s1.union(s2)
    return len(s1.intersection(s2)) / len(union) if union else 0


def precalc_GO(drug_targets, protein_annot, go_graph):
    """
    预计算每个药物的扩展GO术语集合，并基于此计算药物间Jaccard相似性。
    同时构建药物-GO术语矩阵，便于后续分析。
    """
    drug_terms = {}
    for d, genes in drug_targets.items():
        orig_terms = get_go_terms(genes, protein_annot)
        if orig_terms:
            # 扩展GO术语集合（包含所有祖先）
            expanded_terms = expand_go_terms(orig_terms, go_graph)
            drug_terms[d] = expanded_terms

    valid_drugs = list(drug_terms.keys())
    print(f"> Valid drugs with GO terms: {len(valid_drugs)}/{len(drug_targets)}")

    # 构建药物-GO术语矩阵
    matrix = build_drug_term_matrix(drug_terms)
    matrix_output = "data/Simlarity_net/S7-1.herb_go_matrix.csv"
    os.makedirs(os.path.dirname(matrix_output), exist_ok=True)
    matrix.to_csv(matrix_output)
    print(f"> Drug-term matrix saved to {matrix_output}")

    # 计算两两药物间的Jaccard相似性（基于扩展后的GO集合）
    drug_pairs = list(itertools.combinations(valid_drugs, 2))
    jaccard_sim = []
    for d1, d2 in drug_pairs:
        sim = jaccard_similarity(drug_terms[d1], drug_terms[d2])
        jaccard_sim.append([d1, d2, sim])
    output_file = "data/Simlarity_net/S7-2.herb_jaccard_similarity.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["herb1", "herb2", "Jaccard_Similarity"])
        writer.writerows(jaccard_sim)
    print(f"> Jaccard similarity results saved to {output_file}")


def compute_similarity_batch(batch, go_graph):
    """计算一个批次的疾病对相似性（原代码保持不变）"""
    results = []
    sem_sim = lambda t1, t2: similarity.lin(go_graph, t1, t2)  # GO术语相似性计算

    for disease1, disease2, terms1, terms2 in batch:
        try:
            score = term_set.sim_bma(terms1, terms2, sem_sim)
        except Exception as e:
            print(f"Error processing {disease1} vs {disease2}: {str(e)}")
            score = None
        results.append([disease1, disease2, score])

    return results


def process_and_write_batch(batch, go_graph, output_file):
    """处理单个批次并写入文件"""
    results = compute_similarity_batch(batch, go_graph)
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(results)


def compute_disease_similarity(drug_targets, protein_annot, go_graph, output_file, batch_size=1):
    """
    基于GO图计算疾病相似性（并行+分批处理）。
    """
    print("\n> Precomputing disease GO terms...")
    drug_terms = {d: get_go_terms(genes, protein_annot) for d, genes in drug_targets.items()}

    valid_diseases = [d for d, t in drug_terms.items() if t]  # 过滤无GO术语的疾病
    print(f"> Valid diseases with GO terms: {len(valid_diseases)}/{len(drug_targets)}")

    # 生成所有两两疾病组合
    disease_pairs = list(itertools.combinations(valid_diseases, 2))
    print(f"> Total disease pairs to calculate: {len(disease_pairs)}")

    # 创建任务列表
    tasks = [(d1, d2, drug_terms[d1], drug_terms[d2]) for d1, d2 in disease_pairs]

    # 分批处理
    batch_samples = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    print(f"> Total batches: {len(batch_samples)}, Batch size: {batch_size}")

    # 初始化 CSV 文件，写入表头
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["symptom1", "symptom2", "Similarity"])

    # 并行处理
    print("\n> Starting parallel computation...")
    s = time.time()
    Parallel(n_jobs=16)(delayed(process_and_write_batch)(batch, go_graph, output_file) for batch in batch_samples)
    e = time.time()

    print(f"\n> Computation finished in {e - s:.2f} seconds. Results saved to {output_file}")


def main():
    # 1. 下载资源文件
    if not os.path.exists("_resources/go-basic.obo"):
        download.obo()
        download.gaf()

    # 2. 构建GO图（作为GO DAG使用）
    go_graph = graph.from_resource("go-basic")
    similarity.precalc_lower_bounds(go_graph)

    # 3. 加载蛋白注释
    protein_annot = annotation.from_resource("goa_human")
    print(f"Total annotated proteins: {len(protein_annot)}")

    # 4. 读取中药-靶标数据
    drug_targets_file = r'data/S2.HIT_herb_target_data_0412dropna.csv'
    drug_targets = parse_drug_target(drug_targets_file)
    # 读取症状-蛋白数据（需将临床症状-靶标也考虑到计算中）
    disease_gene_file = r'data/S1.TCM_symptom_genes_association_genes_20_add_clinical_symptoms.csv'
    disease_genes = parse_disease_genes(disease_gene_file)

    # 5. 计算中药间Jaccard相似性
    precalc_GO(drug_targets, protein_annot, go_graph)

    # 计算症状间Lin相似性
    output_file = "data/Simlarity_net/S8.symptom_similarity.csv"
    compute_disease_similarity(disease_genes, protein_annot, go_graph, output_file)


if __name__ == "__main__":
    main()
