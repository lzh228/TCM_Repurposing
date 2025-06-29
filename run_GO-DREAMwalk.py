from Random_Walk.generate_embeddings_GO_DREAMwalk import save_embedding_files
from Random_Walk.predict_associations_GO_DREAMwalk import predict_asso

if __name__ == '__main__':
    # 网络数据集
    networkf = 'data/S4.Heterogeneous_network.txt'
    simf = 'data/Simlarity_net/S9.sim_graph.txt'
    nodetypef = 'data/S6.node_types.tsv'

    # 标签数据集
    pairf = 'data/S5-1.dda.tsv'
    clinical_pairs = 'data/S5-3.clinical_pairs.tsv'
    effective_pairs = 'data/S5-2.effective_pairs.tsv'

    # 结果文件
    embeddingf = f"results/GO-DREAMwalk_results/embedding_file_sim_net.pkl"
    modelf = f'results/GO-DREAMwalk_results/clf_sim_net.pkl'
    clinic_predict_file_path = f'results/GO-DREAMwalk_results/clinical_pairs_sim_net.csv'
    Effective_predict_file_path = f'results/GO-DREAMwalk_results/effective_pairs_sim_net.csv'

    save_embedding_files(netf=networkf, sim_netf=simf, outputf=embeddingf,
                         nodetypef=nodetypef, tp_factor=0.3, walk_length=10, num_walks=100, seed=43)

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