- huggingface 的BERT预训练模型（标准大小，不区分大小写）
- BERT's subwords tokenizer

## 抽取式
(1)将json数据转换为bert输入格式，其中使用了greedy_selection函数处理数据集；
(2)将input输入bert中，利用各句子CLS对应的token embedding进行训练；
(3)训练和验证阶段使用loss保存模型；
(4)使用pyrouge计算得分。