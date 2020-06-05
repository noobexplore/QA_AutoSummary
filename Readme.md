# QA_AutoSummary


### 1. 项目背景

该项目为百度比赛项目，该项目简介为：本次比赛主题为**汽车大师问答摘要与推理**。要求使用汽车大师提供的11万条 **技师与用户的多轮对话与诊断建议报告** 数据建立模型，模型需基于**对话文本、用户问题、车型与车系**，输出包含**摘要与推断**的报告文本，综合考验模型的归纳总结与推断能力。

比赛地址为：https://aistudio.baidu.com/aistudio/competition/detail/3

### 2.项目简介

首先该项目选取seq2seq+attention模型作为baseline，模型文件夹对应seq2seq_model_v2，网络架构图如下：

![seq2seq+att](https://github.com/noobexplore/QA_AutoSummary/blob/master/img/Seq2Seq1.jpg)

然后为了解决一下问题:

- OOV问题即未登录词问题。
- 重复生成词问题。

采用PGN网络和coverage机制进行改进，改进的网络架构如下：

![PGN](https://github.com/noobexplore/QA_AutoSummary/blob/master/img/PGN1.jpg)

#### 改动细节

以上网络框架参考如下论文：

[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)

然后参考：

https://github.com/Light2077/QA-Abstract-And-Reasoning

对文件夹PGN_model中layer层中的decoder进行了修改，发现确实能提高很多性能，具体改动可对照PGN_remodel中layer层里的decoder。

#### 运行

在main.py中修改对应的参数即可进行模型的训练和测试，测试采用greedy和beam两种策略。

```python
python main.py
```

### 模型效果

最终提交到比赛平台上达到30.4的ROUGE-L评分，如下图：

![](https://github.com/noobexplore/QA_AutoSummary/blob/master/img/result1.png)
