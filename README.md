# Chinese_Medical_Natural_Language_Processing_Resources_and_Papers

* [Chinese Medical Natural Language Processing Resources](#chinese_medical_natural_language_processing_resources)
  * [中文医疗数据集](#中文医疗数据集)
    * [1.Yidu-S4K：医渡云结构化4K数据集](#1yidu-s4k医渡云结构化4k数据集)
    * [2.Yidu-N7K：医渡云标准化7K数据集](#2yidu-n7k医渡云标准化7k数据集)
    * [3.瑞金医院MMC人工智能辅助构建知识图谱大赛](#3瑞金医院mmc人工智能辅助构建知识图谱大赛)
    * [4.中文医药方面的问答数据集](#4中文医药方面的问答数据集)
    * [5.平安医疗科技疾病问答迁移学习比赛](#5平安医疗科技疾病问答迁移学习比赛)
    * [6.天池“公益AI之星”挑战赛--新冠疫情相似句对判定大赛](#6天池公益ai之星挑战赛--新冠疫情相似句对判定大赛)
  * [中文医疗知识图谱](#中文医学知识图谱)
    * [1.CmeKG](#1cmekg)
  * [开源工具](#开源工具)
    * [分词工具](#分词工具)
      * [PKUSEG](#pkuseg)
  * [友情链接](#友情链接)
* [Chinese Medical Natural Language Processing Papers](#Chinese_Medical_Natural_Language_Processing_Resources_and_Papers)
  * [ACL Papers](https://github.com/FutureForMe/Chinese_Medical_Natural_Language_Processing_Resources_and_Papers/tree/main/ACL_Papers)
  * [EMNLP Papers](https://github.com/FutureForMe/Chinese_Medical_Natural_Language_Processing_Resources_and_Papers/tree/main/EMNLP_Papers)
  * [NAACL Papers](https://github.com/FutureForMe/Chinese_Medical_Natural_Language_Processing_Resources_and_Papers/tree/main/NAACL_Papers)
  * [AAAI Papers](https://github.com/FutureForMe/Chinese_Medical_Natural_Language_Processing_Resources_and_Papers/tree/main/AAAI_Papers)

### *注：医疗NLP论文汇总md文件及PDF在对应的文件夹中。NAACL 2022已更新。

## 中文医疗数据集

### 1.Yidu-S4K：医渡云结构化4K数据集

> Yidu-S4K 数据集源自CCKS 2019 评测任务一，即“面向中文电子病历的命名实体识别”的数据集，包括两个子任务：

> 1）医疗命名实体识别：由于国内没有公开可获得的面向中文电子病历医疗实体识别数据集，本年度保留了医疗命名实体识别任务，对2017年度数据集做了修订，并随任务一同发布。本子任务的数据集包括训练集和测试集。

> 2）医疗实体及属性抽取（跨院迁移）：在医疗实体识别的基础上，对预定义实体属性进行抽取。本任务为迁移学习任务，即在只提供目标场景少量标注数据的情况下，通过其他场景的标注数据及非标注数据进行目标场景的识别任务。本子任务的数据集包括训练集（非目标场景和目标场景的标注数据、各个场景的非标注数据）和测试集（目标场景的标注数据）。

数据集地址：[http://openkg.cn/dataset/yidu-s4k](http://openkg.cn/dataset/yidu-s4k)

### 2.Yidu-N7K：医渡云标准化7K数据集

> 数据描述：Yidu-N4K 数据集源自CHIP 2019 评测任务一，即“临床术语标准化任务”的数据集。

> 临床术语标准化任务是医学统计中不可或缺的一项任务。临床上，关于同一种诊断、手术、药品、检查、化验、症状等往往会有成百上千种不同的写法。标准化（归一）要解决的问题就是为临床上各种不同说法找到对应的标准说法。有了术语标准化的基础，研究人员才可对电子病历进行后续的统计分析。本质上，临床术语标准化任务也是语义相似度匹配任务的一种。但是由于原词表述方式过于多样，单一的匹配模型很难获得很好的效果。

数据集地址：[http://openkg.cn/dataset/yidu-n7k](http://openkg.cn/dataset/yidu-n7k)

### 3.瑞金医院MMC人工智能辅助构建知识图谱大赛

> 赛题描述：本次大赛旨在通过糖尿病相关的教科书、研究论文来做糖尿病文献挖掘并构建糖尿病知识图谱。参赛选手需要设计高准确率，高效的算法来挑战这一科学难题。第一赛季课题为“基于糖尿病临床指南和研究论文的实体标注构建”，第二赛季课题为“基于糖尿病临床指南和研究论文的实体间关系构建”。

数据集地址：[https://tianchi.aliyun.com/competition/entrance/231687/information](https://tianchi.aliyun.com/competition/entrance/231687/information)

### 4.中文医药方面的问答数据集

> 数据描述：该数据集由IEEE中一篇论文中提出，名为：Multi-Scale Attentive Interaction Networks for Chinese Medical Question Answer Selection，他是一个面向中文医疗方向的问答数据集，数量级别达10万级。
> 文件说明：questions.csv：所有的问题及其内容；answers.csv：所有问题的答案；train_candidates.txt， dev_candidates.txt， test_candidates.txt：将上述两个文件进行了拆分。

数据集地址：[https://github.com/zhangsheng93/cMedQA2](https://github.com/zhangsheng93/cMedQA2)

### 5.平安医疗科技疾病问答迁移学习比赛

> 任务描述：本次比赛是chip2019中的评测任务二，由平安医疗科技主办。本次评测任务的主要目标是针对中文的疾病问答数据，进行病种间的迁移学习。具体而言，给定来自5个不同病种的问句对，要求判定两个句子语义是否相同或者相近。所有语料来自互联网上患者真实的问题，并经过了筛选和人工的意图匹配标注。首页说明了相关数据的格式。

数据集地址：[https://www.biendata.xyz/competition/chip2019/](https://www.biendata.xyz/competition/chip2019/) 需注册才能下载

### 6.天池“公益AI之星”挑战赛--新冠疫情相似句对判定大赛

> 赛制说明：比赛主打疫情相关的呼吸领域的真实数据积累，数据粒度更加细化，判定难度相比多科室文本相似度匹配更高，同时问答数据也更具时效性。本着宁缺毋滥的原则，问题的场地限制在20字以内，形成相对规范的句对。要求选手通过自然语义算法和医学知识识别相似问答和无关的问题。相关数据说明参见比赛网址首页。

数据集地址：[https://tianchi.aliyun.com/competition/entrance/231776/information](https://tianchi.aliyun.com/competition/entrance/231776/information) 需注册才能下载


## 中文医学知识图谱

### 1.CMEKG

> 知识图谱简介：CMeKG（Chinese Medical Knowledge Graph）是利用自然语言处理与文本挖掘技术，基于大规模医学文本数据，以人机结合的方式研发的中文医学知识图谱。CMeKG的构建参考了ICD、ATC、SNOMED、MeSH等权威的国际医学标准以及规模庞大、多源异构的临床指南、行业标准、诊疗规范与医学百科等医学文本信息。CMeKG 1.0包括：6310种疾病、19853种药物（西药、中成药、中草药）、1237种诊疗技术及设备的结构化知识描述，涵盖疾病的临床症状、发病部位、药物治疗、手术治疗、鉴别诊断、影像学检查、高危因素、传播途径、多发群体、就诊科室等以及药物的成分、适应症、用法用量、有效期、禁忌证等30余种常见关系类型，CMeKG描述的概念关系实例及属性三元组达100余万。

CMEKG图谱地址：[http://cmekg.pcl.ac.cn/](http://cmekg.pcl.ac.cn/)

## 开源工具

### 分词工具

#### PKUSEG

pkuseg 是由北京大学推出的基于论文PKUSEG: A Toolkit for Multi-Domain Chinese Word Segmentation 的工具包。其简单易用，支持细分领域分词，有效提升了分词准确度。

> pkuseg具有如下几个特点：
> 1.多领域分词。不同于以往的通用中文分词工具，此工具包同时致力于为不同领域的数据提供个性化的预训练模型。根据待分词文本的领域特点，用户可以自由地选择不同的模型。 我们目前支持了新闻领域，网络领域，医药领域，旅游领域，以及混合领域的分词预训练模型。在使用中，如果用户明确待分词的领域，可加载对应的模型进行分词。如果用户无法确定具体领域，推荐使用在混合领域上训练的通用模型。各领域分词样例可参考 example.txt。
> 2.更高的分词准确率。相比于其他的分词工具包，当使用相同的训练数据和测试数据，pkuseg可以取得更高的分词准确率。
> 3.支持用户自训练模型。支持用户使用全新的标注数据进行训练。
> 4.支持词性标注。

项目地址：[https://github.com/lancopku/pkuseg-python](https://github.com/lancopku/pkuseg-python)

## 友情链接

[awesome_Chinese_medical_NLP](https://github.com/GanjinZero/awesome_Chinese_medical_NLP)

[Chinese_medical_NLP](https://github.com/lrs1353281004/Chinese_medical_NLP)











