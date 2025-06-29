# README
***
 # **基于异构蛋白质网络随机游走的中药重定位模型**
***
## 数据集介绍

S1.TCM_symptom_genes_association_genes_20. csv 
症状-蛋白关联数据<sup>[1]</sup>，包含174种症状及其对应的12,601条症状-关联蛋白数据

S2.HIT_herb_target_data_0412dropna.csv
中药-靶标数据<sup>[1]</sup>，包含798种中药及127,759条中药-靶标蛋白关联数据

S3.clinical_symptoms_genes_160symptoms.csv
肝硬化临床症状数据<sup>[1]</sup>，包含160种临床症状及12,646条症状-蛋白关联数据

S4.Heterogeneous_network.txt
中药-蛋白质-症状网络，由S1, S2, S3及蛋白质相互作用网络<sup>[2]</sup>构成，包含19,578个节点和480,674条边

S5-1.dda.tsv
有治疗关联的中药-症状标签数据<sup>[1]</sup>

S5-2.effective_pairs.tsv
临床有效的中药-临床症状标签数据<sup>[1]</sup>

S5-3.clinical_pairs.tsv
临床出现的中药-临床症状标签数据<sup>[1]</sup>

S6.node_types.tsv
中药-蛋白质-症状网络中所有节点的类型数据

S7-1.herb_go_matrix.csv
中药-GO向量矩阵数据，行数据表示每种中药的GO向量表示

S7-2.herb_jaccard_similarity.csv
中药-中药相似性网络数据，由Jaccard Simlarity计算所得

S8.symptom_similarity.csv
症状-症状相似性网络数据，由Lin Simlarity计算所得

S9.sim_graph.txt
相似性网络数据，取S7-2与S8权重排名前60%的相似性边

***
## 相似性网络计算
 - 代码：Simlarity_net_cal.py
 - 描述：使用pygosemsim包自动下载Gene Ontology数据库中的蛋白质-GO术语文件及 GO 术语间的层级关系文件<sup>[1]</sup>。基于GO术语层级图，构建中药-GO向量矩阵，并使用Jaccard Simlarity计算中药-中药相似性分数，使用Lin Simlarity计算症状-症状相似性分数。
 - 输入：数据集S1, S2, S3
 - 输出：数据集S7-1, S7-2, S8
***
## GO-DREAM模型预测中药-症状关联

 - 代码：run_GO-DREAMwalk.py
 - 描述：使用相似性网络数据 S9 与 中药-蛋白质-症状网络数据 S4 构建多层网络，使用DREAMwalk<sup>[3]</sup>在该网络上执行随机游走，生成节点序列后输入到异构Skip-gram模型，学习节点的嵌入向量表示。随后，结合中药-症状标签与嵌入向量训练XGBoost分类器，最终在肝硬化临床数据上对模型进行测试与评估。
 - 输入：数据集S4, S5-1, S5-2, S5-3, S6, S9
 - 输出：节点嵌入向量文件：embedding_file_sim_net.pkl，XGBoost模型参数文件：clf_sim_net.pkl，分别带有临床有效和临床出现标签的中药-临床症状关联分数文件：effective_pairs_sim_net.csv, clinical_pairs_sim_net.csv 
***
## M-RW模型预测中药-症状关联

 - 代码：run_M-RW.py
 - 描述：执行以MetaPath为路径规则的随机游走生成节点序列，输入到异构Skip-gram模型，学习节点的嵌入向量表示。随后，结合中药-症状标签与嵌入向量训练XGBoost分类器，最终在肝硬化临床数据上对模型进行测试与评估。
 - 输入：数据集S4, S5-1, S5-2, S5-3, S6
 - 输出：节点嵌入向量文件：embedding_file_100_100_M4.pkl，XGBoost模型参数文件：clf_M4.pkl, 分别带有临床有效和临床出现标签的中药-临床症状关联分数文件：effective_pairs_M4.csv, clinical_pairs_M4.csv
***
## Rank Aggregation

 - 代码：Rank_Aggregation.py
 - 描述：分别采用Crank、Borda、Dowdall三种Rank Aggregation算法<sup>[2, 4]</sup>，对M-RW和GO-DREAMwalk的预测结果进行聚合，并绘制预测得分排名前k项范围内的Precision和Recall性能变化图。结果表明，Borda算法能很好的融合两个模型，预测性能得到进一步提升。最终生成基于Borda聚合的预测结果文件。
 - 输入：effective_pairs_M4.csv，effective_pairs_sim_net.csv
 - 输出：Border_results.csv
***
## 参考文献
1. *Gan X, Shu Z, Wang X, et al. Network medicine framework reveals generic herb-symptom effectiveness of traditional Chinese medicine[J]. Science Advances, 2023, 9(43): eadh0215.*
2. *Morselli Gysi D, Do Valle Í, Zitnik M, et al. Network medicine framework for identifying drug-repurposing opportunities for COVID-19[J]. Proceedings of the National Academy of Sciences, 2021, 118(19): e2025581118.*
3. *Bang D. Biomedical knowledge graph learning for drug repurposing by extending guilt-by-association to multiple layers[J].*
4. *Spector J, Aldana A, Sebek M, et al. Transformers Enhance the Predictive Power of Network Medicine[J]. Pharmacology and Therapeutics, 2025.*


