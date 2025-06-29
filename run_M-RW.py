from Random_Walk.generate_embeddings_M_RW import save_embedding_files
from Random_Walk.predict_associations_M_RW import predict_asso

# from Random_Walk.generate_embeddings_single_sampling import save_embedding_files
# 单向采样
from Random_Walk.generate_embeddings_control_group import run_random_walk_control_experiments

if __name__ == '__main__':
    # 网络数据集
    networkf = 'data/S4.Heterogeneous_network.txt'
    # 节点类型数据集
    nodetypef = 'data/S6.node_types.tsv'
    # 标签数据集
    pairf = 'data/S5-1.dda.tsv'
    clinical_pairs = 'data/S5-3.clinical_pairs.tsv'
    effective_pairs = 'data/S5-2.effective_pairs.tsv'
    # 结果文件
    embeddingf = f"results/M-RW_results/embedding_file_100_100_M4.pkl"
    modelf = f'results/M-RW_results/clf_M4.pkl'
    clinic_predict_file_path = f'results/M-RW_results/clinical_pairs_M4.csv'
    Effective_predict_file_path = f'results/M-RW_results/effective_pairs_M4.csv'

    # 消融实验
    # embeddingf = f"results/control_embeddings/control_randomwalk_embedding_trial{i}.pkl"
    # modelf = f'results/control_embeddings/temp/clf{i}.pkl'
    # output_dir = 'results/control_embeddings/'

    # 消融实验生成嵌入向量
    # run_random_walk_control_experiments(netf=networkf, output_dir=output_dir, nodetypef=nodetypef)

    # 生成节点嵌入
    save_embedding_files(netf=networkf, outputf=embeddingf, nodetypef=nodetypef, walk_length=100, num_walks=100, seed=43)



    # 训练模型
    predict_asso(embeddingf, pairf, clinic_predict_file_path, modelf, seed=43, valid_ratio=0.1, test_ratio=0.1,
                 train=True)

    # 使用临床数据检测模型性能
    predict_asso(
        embedding_file=embeddingf,
        pair_file=effective_pairs,
        clinic_predict_file_path=Effective_predict_file_path,
        model_checkpoint=modelf,
        seed=43,
        valid_ratio=0,  # 无需验证集
        test_ratio=1.0,  # 所有数据作为测试集
        train=False  # 关闭训练模式
    )
    print('=' * 50)
    # 使用临床数据检测模型性能
    predict_asso(
        embedding_file=embeddingf,
        pair_file=clinical_pairs,
        clinic_predict_file_path=clinic_predict_file_path,
        model_checkpoint=modelf,
        seed=43,
        valid_ratio=0,  # 无需验证集
        test_ratio=1.0,  # 所有数据作为测试集
        train=False  # 关闭训练模式
    )
