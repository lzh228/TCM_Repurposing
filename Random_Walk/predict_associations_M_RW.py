import argparse
import csv
import pickle
import numpy as np
import joblib
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, recall_score
from Random_Walk.utils import set_seed
from sklearn.calibration import CalibratedClassifierCV

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', type=str, required=True)
    parser.add_argument('--pair_file', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_checkpoint', type=str, default='clf.pkl')
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--train', action='store_true', help="启用训练模式")

    return parser.parse_args()


def split_dataset(pair_file, embedding_file, valid_ratio, test_ratio, seed):
    with open(embedding_file, 'rb') as f:
        embedding_dict = pickle.load(f)

    xs, ys, pairs = [], [], []
    with open(pair_file, 'r') as f:
        lines = f.readlines()[1:]

    for line in lines:
        drug, dis, label = line.strip().split('\t')
        xs.append(embedding_dict[drug] - embedding_dict[dis])
        ys.append(int(label))
        pairs.append((drug, dis))

    xs, ys = np.array(xs), np.array(ys)

    x, y, pair_data = {}, {}, {}

    # 处理测试集占比为100%的情况
    if test_ratio >= 1.0:
        x['test'], y['test'], pair_data['test'] = xs, ys, pairs
        x['train'], y['train'], pair_data['train'] = np.array([]), np.array([]), []
        x['valid'], y['valid'], pair_data['valid'] = np.array([]), np.array([]), []
    else:
        # 原有分割逻辑
        (x['train'], x['test'], y['train'], y['test'], pair_data['train'], pair_data['test']) = train_test_split(
            xs, ys, pairs, test_size=test_ratio, random_state=seed, stratify=ys
        )

        if valid_ratio > 0:
            (x['train'], x['valid'], y['train'], y['valid'], pair_data['train'], pair_data['valid']) = train_test_split(
                x['train'], y['train'], pair_data['train'],
                test_size=valid_ratio / (1 - test_ratio),
                random_state=seed,
                stratify=y['train']
            )
        else:
            x['valid'], y['valid'], pair_data['valid'] = [], [], []

    # **交叉检查，确保数据没有泄露**
    if pair_data['train'] is not None and pair_data['test'] is not None:
        assert len(set(pair_data['train']) & set(pair_data['test'])) == 0, "训练集和测试集有重叠，数据泄露！"
    if pair_data['train'] is not None and pair_data['valid'] is not None:
        assert len(set(pair_data['train']) & set(pair_data['valid'])) == 0, "训练集和验证集有重叠，数据泄露！"
    if pair_data['valid'] is not None and pair_data['test'] is not None:
        assert len(set(pair_data['valid']) & set(pair_data['test'])) == 0, "验证集和测试集有重叠，数据泄露！"

    return x, y, pair_data

def return_scores(target_list, pred_list):
    """
    计算并返回模型的多种评估指标：
    1. AUROC
    2. 全局 AUPR
    3. 前1% 预测项的平均精度 (Precision@1%)
    4. 前1% 预测项的召回率 (Recall@1%)
    """
    # 输入检查
    if len(target_list) == 0 or len(pred_list) == 0:
        print("Error: 输入列表为空")
        return None

    # 转换数据类型
    target_array = np.array(target_list)
    pred_array = np.array(pred_list)

    # 计算 AUROC
    auroc = roc_auc_score(target_array, pred_array)

    # 计算全局 AUPR
    average_precision = average_precision_score(target_array, pred_array)

    # 选取前 1% 预测项的索引 (优化排序)
    top_k = max(1, len(pred_array) // 100)  # 确保至少有 1 个
    top_1_percent_idx = np.argpartition(pred_array, -top_k)[-top_k:]
    top_1_percent_idx = top_1_percent_idx[np.argsort(pred_array[top_1_percent_idx])[::-1]]

    # 获取前 1% 样本的真实标签
    top_labels = target_array[top_1_percent_idx]

    # 计算 Precision@1%
    precision_at_1_percent = np.mean(top_labels)  # 计算前 1% 样本中的正样本比例

    # 计算 Recall@1%
    total_positives = np.sum(target_array == 1)  # 统计所有正样本
    true_positives_in_top = np.sum(top_labels == 1)  # 前1% 样本中的正样本数
    recall_at_1_percent = true_positives_in_top / total_positives if total_positives > 0 else 0

    print(f"\nRecall@1%: {recall_at_1_percent:.4f} ({true_positives_in_top}/{total_positives})")
    return {
        'AUROC': auroc,
        'AUPR': average_precision,
        'Precision (Top 1%)': precision_at_1_percent,
        'Recall (Top 1%)': recall_at_1_percent
    }


def predict_asso(embedding_file, pair_file,clinic_predict_file_path, model_checkpoint, seed, valid_ratio, test_ratio, train=True):
    set_seed(seed)

    # 数据集划分
    x, y, pair_data = split_dataset(pair_file, embedding_file, valid_ratio, test_ratio, seed)

    if train:
        best_params = {
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'learning_rate': 0.08,
            'max_depth': 7,
            'subsample': 0.8
        }

        # 计算正负样本比例（假设 y_train 是训练标签）
        train_labels = y['train']
        positive_count = sum(train_labels == 1)
        negative_count = sum(train_labels == 0)
        scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0
        print(f"\n类别权重计算: 正样本={positive_count}, 负样本={negative_count}, scale_pos_weight={scale_pos_weight:.2f}")


        # 取最优模型并在训练集+验证集上训练
        print("\n直接使用最佳参数训练模型模型...")
        best_model = XGBClassifier(
            # **grid_search.best_params_,
            **best_params,
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            n_estimators=500,
            early_stopping_rounds=50,
            random_state=seed
        )

        best_model.fit(
            x['train'], y['train'],
            eval_set=[(x['train'], y['train']), (x['valid'], y['valid'])],  # 监控训练和验证损失
            verbose=False
        )

        # # ========== 绘制损失曲线 ==========
        # evals_result = best_model.evals_result()
        # train_losses = evals_result['validation_0']['logloss']  # 训练集损失
        # valid_losses = evals_result['validation_1']['logloss']  # 验证集损失
        #
        # plt.figure(figsize=(8, 6))
        # plt.plot(train_losses, label="Train Loss", color='blue')
        # plt.plot(valid_losses, label="Validation Loss", color='red')
        # plt.xlabel("Number of Rounds")
        # plt.ylabel("Log Loss")
        # plt.title("Training vs. Validation Loss")
        # plt.legend()
        # plt.show()

        # 进行概率校准
        calibrated_clf = CalibratedClassifierCV(best_model, cv="prefit", method="isotonic")
        calibrated_clf.fit(x['valid'], y['valid'])

        joblib.dump(calibrated_clf, model_checkpoint)
        print(f"\n已保存调优后的模型: {model_checkpoint}")

    # 测试模式
    print("\n开始测试...")
    best_clf = joblib.load(model_checkpoint)
    preds = best_clf.predict_proba(np.array(x['test']))[:, 1]
    scores = return_scores(y['test'], preds)

    print(f"\nTEST集 | AUROC: {scores['AUROC'] * 100:.2f}% | AUPR: {scores['AUPR'] * 100:.2f}% | "
          f"Precision (Top 1%): {scores['Precision (Top 1%)']:.4f} | Recall(Top 1%): {scores['Recall (Top 1%)']:.4f}")

    # 在预测代码块末尾添加以下代码（替换原有的txt保存部分）：
    if test_ratio==1.0:
        with open(clinic_predict_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["herb", "symptom", "label","score"])  # CSV头部
            for idx, ((drug, disease), pred) in enumerate(zip(pair_data['test'], preds)):
                true_label = y['test'][idx]  # 从标签数组中按索引获取真实标签

                writer.writerow([drug, disease, true_label, f"{pred:.4f}"])
        print("预测结果已保存到 predictions.csv")

    print('=' * 50)

if __name__ == '__main__':
    args = parse_args()
    predict_asso(
        embedding_file=args.embedding_file,
        pair_file=args.pair_file,
        model_checkpoint=args.model_checkpoint,
        seed=args.seed,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        train=args.train
    )
