import os
import pandas as pd
import matplotlib.pyplot as plt
from tools import combine_scores as combiner


def calculate_ml_metrics(df, score_col, label_col, max_k=1000):
    df = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    precision_list = []
    recall_list = []

    total_positives = df[label_col].sum()

    for k in range(1, max_k+1):
        top_k = df.iloc[:k]
        tp = top_k[label_col].sum()
        precision = tp / k
        recall = tp / total_positives if total_positives > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)

    return {
        'precision': precision_list,
        'recall': recall_list
    }

def plot_precision(metrics, colors, labels,mak_k, save_file=None):
    plt.figure()
    for metric, color, label in zip(metrics, colors, labels):
        plt.plot(range(1, mak_k+1), metric['precision'], color=color, label=label)
    plt.xlabel('Top-k')
    plt.ylabel('Precision')
    plt.legend(prop={'size': 16})
    if save_file:
        plt.savefig(save_file, dpi=600)
    plt.show()

def plot_recall(metrics, colors, labels, mak_k, save_file=None):
    plt.figure()
    for metric, color, label in zip(metrics, colors, labels):
        plt.plot(range(1, mak_k+1), metric['recall'], color=color, label=label)
    plt.xlabel('Top-k')
    plt.ylabel('Recall')
    plt.legend(prop={'size': 16})
    if save_file:
        plt.savefig(save_file, dpi=600)
    plt.show()

# def plot_ml_group(metrics, colors, labels, mak_k, saveDir):
#     plot_precision(metrics, colors, labels, mak_k,save_file=saveDir + "precision_clinical.pdf" if saveDir else None)
#     plot_recall(metrics, colors, labels, mak_k, save_file=saveDir + "recall_clinical.pdf" if saveDir else None)
def plot_ml_group(metrics, colors, labels, mak_k, save_file=None):
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))  # 两行一列

    # Precision 子图
    for metric, color, label in zip(metrics, colors, labels):
        axes[0].plot(range(1, mak_k + 1), metric['precision'], color=color, label=label)
    axes[0].set_xlabel('Top-k')
    axes[0].set_ylabel('Precision')
    axes[0].legend(prop={'size': 15})

    # Recall 子图
    for metric, color, label in zip(metrics, colors, labels):
        axes[1].plot(range(1, mak_k + 1), metric['recall'], color=color, label=label)
    axes[1].set_xlabel('Top-k')
    axes[1].set_ylabel('Recall')
    axes[1].legend(prop={'size': 15})

    plt.tight_layout()

    if save_file:
        plt.savefig(save_file, dpi=600)
    plt.show()


if __name__ == '__main__':
    # === 读取两个csv预测结果文件 ===

    df1 = pd.read_csv(r"results/GO-DREAMwalk_results/effective_pairs_sim_net.csv")
    df2 = pd.read_csv(r"results/M-RW_results/effective_pairs_M4.csv")

    # === 组合算法 ===
    scores_borda = combiner.borda_count(df1, df2, 'score', 'herb', 'symptom')
    scores_dawdall = combiner.dawdall_count(df1, df2, 'score', 'herb', 'symptom')
    scores_crank = combiner.crank_count(df1, df2, 'score', 'herb', 'symptom', p=4)

    # === 格式统一化 ===
    df1['method'] = 'GO-DREAMwalk'
    df2['method'] = 'M-RW'
    scores_borda['method'] = 'Borda'
    scores_dawdall['method'] = 'Dawdall'
    scores_crank['method'] = 'CRank'

    scores_borda.rename(columns={'Total_Borda': 'score', 'label_df1': 'label'}, inplace=True)
    scores_dawdall.rename(columns={'Total_Dawdall': 'score', 'label_df1': 'label'}, inplace=True)
    scores_crank.rename(columns={'Total_CRank': 'score', 'label_df1': 'label'}, inplace=True)

    # === 保存Borda预测结果为CSV文件 ===
    # 确保输出目录存在
    output_dir = "results/Borda_results/"
    os.makedirs(output_dir, exist_ok=True)

    # 保存Borda结果
    borda_output_file = os.path.join(output_dir, "borda_predictions_effective.csv")
    scores_borda.to_csv(borda_output_file, index=False)
    print(f"Borda预测结果已保存到: {borda_output_file}")

    # === 计算所有方法的Precision/Recall列表 ===
    dataframes = [df1, df2, scores_borda, scores_dawdall, scores_crank]
    max_k = int(len(df1)*0.01)
    print(max_k)
    metrics = [calculate_ml_metrics(df, 'score', 'label', max_k= max_k ) for df in dataframes]
    colors = ['black', 'red', 'blue', 'purple', 'green']
    labels = ['GO-DREAMwalk', 'M-RW', 'Borda', 'Dowdall', 'CRank']

    # # === 绘图 ===
    # # plot_ml_group(metrics, colors, labels, int(len(df1)*0.01),"results/")
    plot_ml_group(metrics, colors, labels, int(len(df1)*0.01), save_file="precision_recall_effective.pdf")

    # # 获取 Borda 的指标
    # borda_idx = labels.index('Borda')  # 找到 Borda 在列表中的位置
    # borda_metrics = metrics[borda_idx]

    # # 获取 Borda 的指标
    # borda_idx = labels.index('Borda')  # 找到 Borda 在列表中的位置
    # borda_metrics = metrics[borda_idx]
    #
    # # 输出 Top-k 的 Precision 和 Recall
    # top_k_index = max_k - 1  # 因为索引从0开始
    # precision_at_k = borda_metrics['precision'][top_k_index]
    # recall_at_k = borda_metrics['recall'][top_k_index]
    #
    # print(f"Borda at Top-{max_k}:")
    # print(f"  Precision = {precision_at_k:.4f}")
    # print(f"  Recall    = {recall_at_k:.4f}")


