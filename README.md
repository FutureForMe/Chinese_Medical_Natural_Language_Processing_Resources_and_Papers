# Chinese_Medical_Natural_Language_Processing_Resources_and_Papers

* [Chinese Medical Natural Language Processing_Resources](#chinese_medical_natural_language_processing_resources)
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
* [Medical_Natural_Language_Processing_Papers](#medical_natural_language_processing_papers)
  * [1.ACL 2021](#1acl-2021)
  * [2.NAACL 2021](#2naacl-2021)
  * [3.AAAI 2021](#3aaai-2021)
  * [4.AAAI 2020](#4aaai-2020)
  * [5.EMNLP 2021](#5emnlp-2021)
  * [6.EMNLP 2020](#6emnlp-2020)


# Chinese_Medical_Natural_Language_Processing_Resources

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



# Medical_Natural_Language_Processing_Papers

医学自然语言处理相关论文汇总，目前主要汇总了ACL2021、NAACL2021、AAAI2021、AAAI2020、ACL2020和EMNLP2020,  EMNLP2021等相关会议论文整理后续有时间还会持续更新。

## 1.ACL 2021
### 问题理解

**A Gradually Soft Multi-Task and Data-Augmented Approach to Medical Question Understanding**

一种逐渐软的多任务和数据增强的医学问题理解方法

论文地址：[https://aclanthology.org/2021.acl-long.119/](https://aclanthology.org/2021.acl-long.119/)

摘要：

Users of medical question answering systems often submit long and detailed questions, making it hard to achieve high recall in answer retrieval. To alleviate this problem, we propose a novel Multi-Task Learning (MTL) method with data augmentation for medical question understanding. We first establish an equivalence between the tasks of question summarization and Recognizing Question Entailment (RQE) using their definitions in the medical domain. Based on this equivalence, we propose a data augmentation algorithm to use just one dataset to optimize for both tasks, with a weighted MTL loss. We introduce gradually soft parameter-sharing: a constraint for decoder parameters to be close, that is gradually loosened as we move to the highest layer. We show through ablation studies that our proposed novelties improve performance. Our method outperforms existing MTL methods across 4 datasets of medical question pairs, in ROUGE scores, RQE accuracy and human evaluation. Finally, we show that our method fares better than single-task learning under 4 low-resource settings.



### 报告生成

**Competence-based Multimodal Curriculum Learning for Medical Report Generation**

用于医学报告生成的基于能力的多模式课程学习

论文地址：[https://aclanthology.org/2021.acl-long.234/](https://aclanthology.org/2021.acl-long.234/)

摘要：

Medical report generation task, which targets to produce long and coherent descriptions of medical images, has attracted growing research interests recently. Different from the general image captioning tasks, medical report generation is more challenging for data-driven neural models. This is mainly due to 1) the serious data bias and 2) the limited medical data. To alleviate the data bias and make best use of available data, we propose a Competence-based Multimodal Curriculum Learning framework (CMCL). Specifically, CMCL simulates the learning process of radiologists and optimizes the model in a step by step manner. Firstly, CMCL estimates the difficulty of each training instance and evaluates the competence of current model; Secondly, CMCL selects the most suitable batch of training instances considering current model competence. By iterating above two steps, CMCL can gradually improve the model’s performance. The experiments on the public IU-Xray and MIMIC-CXR datasets show that CMCL can be incorporated into existing models to improve their performance.





**Writing by Memorizing: Hierarchical Retrieval-based Medical Report Generation**

通过记忆写作：基于层次检索的医学报告生成

论文地址：[https://aclanthology.org/2021.acl-long.387/](https://aclanthology.org/2021.acl-long.387/)

摘要：

Medical report generation is one of the most challenging tasks in medical image analysis. Although existing approaches have achieved promising results, they either require a predefined template database in order to retrieve sentences or ignore the hierarchical nature of medical report generation. To address these issues, we propose MedWriter that incorporates a novel hierarchical retrieval mechanism to automatically extract both report and sentence-level templates for clinically accurate report generation. MedWriter first employs the Visual-Language Retrieval (VLR) module to retrieve the most relevant reports for the given images. To guarantee the logical coherence between generated sentences, the Language-Language Retrieval (LLR) module is introduced to retrieve relevant sentences based on the previous generated description. At last, a language decoder fuses image features and features from retrieved reports and sentences to generate meaningful medical reports. We verified the effectiveness of our model by automatic evaluation and human evaluation on two datasets, i.e., Open-I and MIMIC-CXR.



### 预训练模型

**SMedBERT: A Knowledge-Enhanced Pre-trained Language Model with Structured Semantics for Medical Text Mining**

SMedBERT：用于医学文本挖掘的具有结构化语义的知识增强型预训练语言模型

论文地址：[https://aclanthology.org/2021.acl-long.457/](https://aclanthology.org/2021.acl-long.457/)

摘要：

Recently, the performance of Pre-trained Language Models (PLMs) has been significantly improved by injecting knowledge facts to enhance their abilities of language understanding. For medical domains, the background knowledge sources are especially useful, due to the massive medical terms and their complicated relations are difficult to understand in text. In this work, we introduce SMedBERT, a medical PLM trained on large-scale medical corpora, incorporating deep structured semantic knowledge from neighbours of linked-entity. In SMedBERT, the mention-neighbour hybrid attention is proposed to learn heterogeneous-entity information, which infuses the semantic representations of entity types into the homogeneous neighbouring entity structure. Apart from knowledge integration as external features, we propose to employ the neighbors of linked-entities in the knowledge graph as additional global contexts of text mentions, allowing them to communicate via shared neighbors, thus enrich their semantic representations. Experiments demonstrate that SMedBERT significantly outperforms strong baselines in various knowledge-intensive Chinese medical tasks. It also improves the performance of other tasks such as question answering, question matching and natural language inference.



### 命名实体识别和规范化

**An End-to-End Progressive Multi-Task Learning Framework for Medical Named Entity Recognition and Normalization**

用于医学命名实体识别和规范化的端到端渐进式多任务学习框架

论文地址：[https://aclanthology.org/2021.acl-long.485/](https://aclanthology.org/2021.acl-long.485/)

摘要：

Medical named entity recognition (NER) and normalization (NEN) are fundamental for constructing knowledge graphs and building QA systems. Existing implementations for medical NER and NEN are suffered from the error propagation between the two tasks. The mispredicted mentions from NER will directly influence the results of NEN. Therefore, the NER module is the bottleneck of the whole system. Besides, the learnable features for both tasks are beneficial to improving the model performance. To avoid the disadvantages of existing models and exploit the generalized representation across the two tasks, we design an end-to-end progressive multi-task learning model for jointly modeling medical NER and NEN in an effective way. There are three level tasks with progressive difficulty in the framework. The progressive tasks can reduce the error propagation with the incremental task settings which implies the lower level tasks gain the supervised signals other than errors from the higher level tasks to improve their performances. Besides, the context features are exploited to enrich the semantic information of entity mentions extracted by NER. The performance of NEN profits from the enhanced entity mention features. The standard entities from knowledge bases are introduced into the NER module for extracting corresponding entity mentions correctly. The empirical results on two publicly available medical literature datasets demonstrate the superiority of our method over nine typical methods.





**A Neural Transition-based Joint Model for Disease Named Entity Recognition and Normalization**

基于神经转移的疾病命名实体识别和归一化联合模型

论文地址：[https://aclanthology.org/2021.acl-long.219/](https://aclanthology.org/2021.acl-long.219/)

摘要：

Disease is one of the fundamental entities in biomedical research. Recognizing such entities from biomedical text and then normalizing them to a standardized disease vocabulary offer a tremendous opportunity for many downstream applications. Previous studies have demonstrated that joint modeling of the two sub-tasks has superior performance than the pipelined counterpart. Although the neural joint model based on multi-task learning framework has achieved state-of-the-art performance, it suffers from the boundary inconsistency problem due to the separate decoding procedures. Moreover, it ignores the rich information (e.g., the text surface form) of each candidate concept in the vocabulary, which is quite essential for entity normalization. In this work, we propose a neural transition-based joint model to alleviate these two issues. We transform the end-to-end disease recognition and normalization task as an action sequence prediction task, which not only jointly learns the model with shared representations of the input, but also jointly searches the output by state transitions in one search space. Moreover, we introduce attention mechanisms to take advantage of the text surface form of each candidate concept for better normalization performance. Experimental results conducted on two publicly available datasets show the effectiveness of the proposed method.



### 关系抽取与知识增强

**Joint Biomedical Entity and Relation Extraction with Knowledge-Enhanced Collective Inference**

联合生物医学实体和关系提取与知识增强的集体推理

论文地址：[https://aclanthology.org/2021.acl-long.488/](https://aclanthology.org/2021.acl-long.488/)

摘要：

Compared to the general news domain, information extraction (IE) from biomedical text requires much broader domain knowledge. However, many previous IE methods do not utilize any external knowledge during inference. Due to the exponential growth of biomedical publications, models that do not go beyond their fixed set of parameters will likely fall behind. Inspired by how humans look up relevant information to comprehend a scientific text, we present a novel framework that utilizes external knowledge for joint entity and relation extraction named KECI (Knowledge-Enhanced Collective Inference). Given an input text, KECI first constructs an initial span graph representing its initial understanding of the text. It then uses an entity linker to form a knowledge graph containing relevant background knowledge for the the entity mentions in the text. To make the final predictions, KECI fuses the initial span graph and the knowledge graph into a more refined graph using an attention mechanism. KECI takes a collective approach to link mention spans to entities by integrating global relational information into local representations using graph convolutional networks. Our experimental results show that the framework is highly effective, achieving new state-of-the-art results in two different benchmark datasets: BioRelEx (binding interaction detection) and ADE (adverse drug event extraction). For example, KECI achieves absolute improvements of 4.59% and 4.91% in F1 scores over the state-of-the-art on the BioRelEx entity and relation extraction tasks



### 信息抽取

**Fine-grained Information Extraction from Biomedical Literature based on Knowledge-enriched Abstract Meaning Representation**

基于知识丰富的抽象意义表示的生物医学文献细粒度信息提取

论文地址：[https://aclanthology.org/2021.acl-long.489/](https://aclanthology.org/2021.acl-long.489/)

摘要：

Biomedical Information Extraction from scientific literature presents two unique and non-trivial challenges. First, compared with general natural language texts, sentences from scientific papers usually possess wider contexts between knowledge elements. Moreover, comprehending the fine-grained scientific entities and events urgently requires domain-specific background knowledge. In this paper, we propose a novel biomedical Information Extraction (IE) model to tackle these two challenges and extract scientific entities and events from English research papers. We perform Abstract Meaning Representation (AMR) to compress the wide context to uncover a clear semantic structure for each complex sentence. Besides, we construct the sentence-level knowledge graph from an external knowledge base and use it to enrich the AMR graph to improve the model’s understanding of complex scientific concepts. We use an edge-conditioned graph attention network to encode the knowledge-enriched AMR graph for biomedical IE tasks. Experiments on the GENIA 2011 dataset show that the AMR and external knowledge have contributed 1.8% and 3.0% absolute F-score gains respectively. In order to evaluate the impact of our approach on real-world problems that involve topic-specific fine-grained knowledge elements, we have also created a new ontology and annotated corpus for entity and event extraction for the COVID-19 scientific literature, which can serve as a new benchmark for the biomedical IE community.



### 实体链接

**Learning Domain-Specialised Representations for Cross-Lingual Biomedical Entity Linking**

跨语言生物医学实体链接的学习领域专业表示

论文地址：[https://aclanthology.org/2021.acl-short.72/](https://aclanthology.org/2021.acl-short.72/)

摘要：

Injecting external domain-specific knowledge (e.g., UMLS) into pretrained language models (LMs) advances their capability to handle specialised in-domain tasks such as biomedical entity linking (BEL). However, such abundant expert knowledge is available only for a handful of languages (e.g., English). In this work, by proposing a novel cross-lingual biomedical entity linking task (XL-BEL) and establishing a new XL-BEL benchmark spanning 10 typologically diverse languages, we first investigate the ability of standard knowledge-agnostic as well as knowledge-enhanced monolingual and multilingual LMs beyond the standard monolingual English BEL task. The scores indicate large gaps to English performance. We then address the challenge of transferring domain-specific knowledge in resource-rich languages to resource-poor ones. To this end, we propose and evaluate a series of cross-lingual transfer methods for the XL-BEL task, and demonstrate that general-domain bitext helps propagate the available English knowledge to languages with little to no in-domain data. Remarkably, we show that our proposed domain-specific transfer methods yield consistent gains across all target languages, sometimes up to 20 Precision@1 points, without any in-domain knowledge in the target language, and without any in-domain parallel data.



### 对话生成

**On the Generation of Medical Dialogs for COVID-19**

关于 COVID-19 医疗对话的生成

论文地址：[https://aclanthology.org/2021.acl-short.112/](https://aclanthology.org/2021.acl-short.112/)

摘要：

Under the pandemic of COVID-19, people experiencing COVID19-related symptoms have a pressing need to consult doctors. Because of the shortage of medical professionals, many people cannot receive online consultations timely. To address this problem, we aim to develop a medical dialog system that can provide COVID19-related consultations. We collected two dialog datasets – CovidDialog – (in English and Chinese respectively) containing conversations between doctors and patients about COVID-19. While the largest of their kind, these two datasets are still relatively small compared with general-domain dialog datasets. Training complex dialog generation models on small datasets bears high risk of overfitting. To alleviate overfitting, we develop a multi-task learning approach, which regularizes the data-deficient dialog generation task with a masked token prediction task. Experiments on the CovidDialog datasets demonstrate the effectiveness of our approach. We perform both human evaluation and automatic evaluation of dialogs generated by our method. Results show that the generated responses are promising in being doctor-like, relevant to conversation history, clinically informative and correct. The code and the data are available at https://github.com/UCSD-AI4H/COVID-Dialogue.



### 话语关系分类

**Entity Enhancement for Implicit Discourse Relation Classification in the Biomedical Domain**

生物医学领域隐式话语关系分类的实体增强

论文地址：[https://aclanthology.org/2021.acl-short.116/](https://aclanthology.org/2021.acl-short.116/)

摘要：

Implicit discourse relation classification is a challenging task, in particular when the text domain is different from the standard Penn Discourse Treebank (PDTB; Prasad et al., 2008) training corpus domain (Wall Street Journal in 1990s). We here tackle the task of implicit discourse relation classification on the biomedical domain, for which the Biomedical Discourse Relation Bank (BioDRB; Prasad et al., 2011) is available. We show that entity information can be used to improve discourse relational argument representation. In a first step, we show that explicitly marked instances that are content-wise similar to the target relations can be used to achieve good performance in the cross-domain setting using a simple unsupervised voting pipeline. As a further step, we show that with the linked entity information from the first step, a transformer which is augmented with entity-related information (KBERT; Liu et al., 2020) sets the new state of the art performance on the dataset, outperforming the large pre-trained BioBERT (Lee et al., 2020) model by 2% points.



### 表示学习

**Attentive Multiview Text Representation for Differential Diagnosis**

用于鉴别诊断的注意力集中的多视图文本表示

论文地址：[https://aclanthology.org/2021.acl-short.128/](https://aclanthology.org/2021.acl-short.128/)

摘要：

We present a text representation approach that can combine different views (representations) of the same input through effective data fusion and attention strategies for ranking purposes. We apply our model to the problem of differential diagnosis, which aims to find the most probable diseases that match with clinical descriptions of patients, using data from the Undiagnosed Diseases Network. Our model outperforms several ranking approaches (including a commercially-supported system) by effectively prioritizing and combining representations obtained from traditional and recent text representation techniques. We elaborate on several aspects of our model and shed light on its improved performance.


### 推理

**MedNLI Is Not Immune: Natural Language Inference Artifacts in the Clinical Domain**

MedNLI 不是免疫的：临床领域的自然语言推理工件

论文地址：[https://aclanthology.org/2021.acl-short.129/](https://aclanthology.org/2021.acl-short.129/)

摘要：

Crowdworker-constructed natural language inference (NLI) datasets have been found to contain statistical artifacts associated with the annotation process that allow hypothesis-only classifiers to achieve better-than-random performance (CITATION). We investigate whether MedNLI, a physician-annotated dataset with premises extracted from clinical notes, contains such artifacts (CITATION). We find that entailed hypotheses contain generic versions of specific concepts in the premise, as well as modifiers related to responsiveness, duration, and probability. Neutral hypotheses feature conditions and behaviors that co-occur with, or cause, the condition(s) in the premise. Contradiction hypotheses feature explicit negation of the premise and implicit negation via assertion of good health. Adversarial filtering demonstrates that performance degrades when evaluated on the *difficult* subset. We provide partition information and recommendations for alternative dataset construction strategies for knowledge-intensive domains.


## 2.NAACL 2021
### 本体

The Biomaterials Annotator: a system for ontology-based concept annotation of biomaterials text

论文地址：[https://aclanthology.org/2021.sdp-1.5/](https://aclanthology.org/2021.sdp-1.5/)
### 疾病分类

Towards BERT-based Automatic ICD Coding: Limitations and Opportunities

论文地址：[https://aclanthology.org/2021.bionlp-1.6/](https://aclanthology.org/2021.bionlp-1.6/)

### 小样本学习

Scalable Few-Shot Learning of Robust Biomedical Name Representations

论文地址：[https://aclanthology.org/2021.bionlp-1.3/](https://aclanthology.org/2021.bionlp-1.3/)

### 规范化

Triplet-Trained Vector Space and Sieve-Based Search Improve Biomedical Concept Normalization

论文地址：[https://aclanthology.org/2021.bionlp-1.2/](https://aclanthology.org/2021.bionlp-1.2/)

### 预训练模型

UmlsBERT: Clinical Domain Knowledge Augmentation of Contextual Embeddings Using the Unified Medical Language System Metathesaurus

论文地址：[https://aclanthology.org/2021.naacl-main.139/](https://aclanthology.org/2021.naacl-main.139/)

Self-Alignment Pretraining for Biomedical Entity Representations

论文地址：[https://aclanthology.org/2021.naacl-main.334/](https://aclanthology.org/2021.naacl-main.334/)

Are we there yet? Exploring clinical domain knowledge of BERT models

论文地址：[https://aclanthology.org/2021.bionlp-1.5/](https://aclanthology.org/2021.bionlp-1.5/)

Stress Test Evaluation of Biomedical Word Embeddings

论文地址：[https://aclanthology.org/2021.bionlp-1.13/](https://aclanthology.org/2021.bionlp-1.13/)

BioELECTRA:Pretrained Biomedical text Encoder using Discriminators

论文地址：[https://aclanthology.org/2021.bionlp-1.16/](https://aclanthology.org/2021.bionlp-1.16/)

Improving Biomedical Pretrained Language Models with Knowledge

论文地址：[https://aclanthology.org/2021.bionlp-1.20/](https://aclanthology.org/2021.bionlp-1.20/)

EntityBERT: Entity-centric Masking Strategy for Model Pretraining for the Clinical Domain

论文地址：[https://aclanthology.org/2021.bionlp-1.21/](https://aclanthology.org/2021.bionlp-1.21/)

ChicHealth @ MEDIQA 2021: Exploring the limits of pre-trained seq2seq models for medical summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.29/](https://aclanthology.org/2021.bionlp-1.29/)

### 命名实体识别

Exploring Word Segmentation and Medical Concept Recognition for Chinese Medical Texts

论文地址：[https://aclanthology.org/2021.bionlp-1.23/](https://aclanthology.org/2021.bionlp-1.23/)

### 因果关系

Are we there yet? Exploring clinical domain knowledge of BERT models

论文地址：[https://aclanthology.org/2021.bionlp-1.5/](https://aclanthology.org/2021.bionlp-1.5/)

### 关系抽取

Improving BERT Model Using Contrastive Learning for Biomedical Relation Extraction

论文地址：[https://aclanthology.org/2021.bionlp-1.1/](https://aclanthology.org/2021.bionlp-1.1/)

### 实体链接

Clustering-based Inference for Biomedical Entity Linking

论文地址：[https://aclanthology.org/2021.naacl-main.205/](https://aclanthology.org/2021.naacl-main.205/)

End-to-end Biomedical Entity Linking with Span-based Dictionary Matching

论文地址：[https://aclanthology.org/2021.bionlp-1.18/](https://aclanthology.org/2021.bionlp-1.18/)

Word-Level Alignment of Paper Documents with their Electronic Full-Text Counterparts

论文地址：[https://aclanthology.org/2021.bionlp-1.19/](https://aclanthology.org/2021.bionlp-1.19/)

### 语言模型

BioM-Transformers: Building Large Biomedical Language Models with BERT, ALBERT and ELECTRA

论文地址：[https://aclanthology.org/2021.bionlp-1.24/](https://aclanthology.org/2021.bionlp-1.24/)

Semi-Supervised Language Models for Identification of Personal Health Experiential from Twitter Data: A Case for Medication Effects

论文地址：[https://aclanthology.org/2021.bionlp-1.25/](https://aclanthology.org/2021.bionlp-1.25/)

Assertion Detection in Clinical Notes: Medical Language Models to the Rescue?

论文地址：[https://aclanthology.org/2021.nlpmc-1.5/](https://aclanthology.org/2021.nlpmc-1.5/)

### 摘要生成

UETrice at MEDIQA 2021: A Prosper-thy-neighbour Extractive Multi-document Summarization Model

论文地址：[https://aclanthology.org/2021.bionlp-1.36/](https://aclanthology.org/2021.bionlp-1.36/)

IBMResearch at MEDIQA 2021: Toward Improving Factual Correctness of Radiology Report Abstractive Summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.35/](https://aclanthology.org/2021.bionlp-1.35/)

Optum at MEDIQA 2021: Abstractive Summarization of Radiology Reports using simple BART Finetuning

论文地址：[https://aclanthology.org/2021.bionlp-1.32/](https://aclanthology.org/2021.bionlp-1.32/)

MNLP at MEDIQA 2021: Fine-Tuning PEGASUS for Consumer Health Question Summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.37/](https://aclanthology.org/2021.bionlp-1.37/)

UETfishes at MEDIQA 2021: Standing-on-the-Shoulders-of-Giants Model for Abstractive Multi-answer Summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.38/](https://aclanthology.org/2021.bionlp-1.38/)

Towards Automating Medical Scribing : Clinic Visit Dialogue2Note Sentence Alignment and Snippet Summarization

论文地址：[https://aclanthology.org/2021.nlpmc-1.2/](https://aclanthology.org/2021.nlpmc-1.2/)

paht_nlp @ MEDIQA 2021: Multi-grained Query Focused Multi-Answer Summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.10/](https://aclanthology.org/2021.bionlp-1.10/)

### 事件抽取

Counterfactual Supporting Facts Extraction for Explainable Medical Record Based Diagnosis with Graph Network

论文地址：[https://aclanthology.org/2021.naacl-main.156/](https://aclanthology.org/2021.naacl-main.156/)

### 迁移学习

UCSD-Adobe at MEDIQA 2021: Transfer Learning and Answer Sentence Selection for Medical Summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.28/](https://aclanthology.org/2021.bionlp-1.28/)

SB_NITK at MEDIQA 2021: Leveraging Transfer Learning for Question Summarization in Medical Domain

论文地址：[https://aclanthology.org/2021.bionlp-1.31/](https://aclanthology.org/2021.bionlp-1.31/)

NLM at MEDIQA 2021: Transfer Learning-based Approaches for Consumer Question and Multi-Answer Summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.34/](https://aclanthology.org/2021.bionlp-1.34/)

### 数据集

emrKBQA: A Clinical Knowledge-Base Question Answering Dataset

论文地址：[https://aclanthology.org/2021.bionlp-1.7/](https://aclanthology.org/2021.bionlp-1.7/)

### 多模态

QIAI at MEDIQA 2021: Multimodal Radiology Report Summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.33/](https://aclanthology.org/2021.bionlp-1.33/)

### 对话

Overview of the MEDIQA 2021 Shared Task on Summarization in the Medical Domain

论文地址：[https://aclanthology.org/2021.bionlp-1.8/](https://aclanthology.org/2021.bionlp-1.8/)

WBI at MEDIQA 2021: Summarizing Consumer Health Questions with Generative Transformers

论文地址：[https://aclanthology.org/2021.bionlp-1.9/](https://aclanthology.org/2021.bionlp-1.9/)

Gathering Information and Engaging the User ComBot: A Task-Based, Serendipitous Dialog Model for Patient-Doctor Interactions

论文地址：[https://aclanthology.org/2021.nlpmc-1.3/](https://aclanthology.org/2021.nlpmc-1.3/)

Extracting Appointment Spans from Medical Conversations

论文地址：[https://aclanthology.org/2021.nlpmc-1.6/](https://aclanthology.org/2021.nlpmc-1.6/)

Building blocks of a task-oriented dialogue system in the healthcare domain

论文地址：[https://aclanthology.org/2021.nlpmc-1.7/](https://aclanthology.org/2021.nlpmc-1.7/)

Medically Aware GPT-3 as a Data Generator for Medical Dialogue Summarization

论文地址：[https://aclanthology.org/2021.nlpmc-1.9/](https://aclanthology.org/2021.nlpmc-1.9/)

### 文本生成

BBAEG: Towards BERT-based Biomedical Adversarial Example Generation for Text Classification

论文地址：[https://aclanthology.org/2021.naacl-main.423/](https://aclanthology.org/2021.naacl-main.423/)

### 问答

NCUEE-NLP at MEDIQA 2021: Health Question Summarization Using PEGASUS Transformers

论文地址：[https://aclanthology.org/2021.bionlp-1.30/](https://aclanthology.org/2021.bionlp-1.30/)

damo_nlp at MEDIQA 2021: Knowledge-based Preprocessing and Coverage-oriented Reranking for Medical Question Summarization

论文地址：[https://aclanthology.org/2021.bionlp-1.12/](https://aclanthology.org/2021.bionlp-1.12/)

### 表示学习

Word centrality constrained representation for keyphrase extraction

论文地址：[https://aclanthology.org/2021.bionlp-1.17/](https://aclanthology.org/2021.bionlp-1.17/)

### Others

Contextual explanation rules for neural clinical classifiers

论文地址：[https://aclanthology.org/2021.bionlp-1.22/](https://aclanthology.org/2021.bionlp-1.22/)

Context-aware query design combines knowledge and data for efficient reading and reasoning

论文地址：[https://aclanthology.org/2021.bionlp-1.26/](https://aclanthology.org/2021.bionlp-1.26/)

Measuring the relative importance of full text sections for information retrieval from scientific literature.

论文地址：[https://aclanthology.org/2021.bionlp-1.27/](https://aclanthology.org/2021.bionlp-1.27/)

Automatic Speech-Based Checklist for Medical Simulations

论文地址：[https://aclanthology.org/2021.nlpmc-1.4/](https://aclanthology.org/2021.nlpmc-1.4/)

Joint Summarization-Entailment Optimization for Consumer Health Question Understanding

论文地址：[https://aclanthology.org/2021.nlpmc-1.8/](https://aclanthology.org/2021.nlpmc-1.8/)

Detecting Anatomical and Functional Connectivity Relations in Biomedical Literature via Language Representation Models

论文地址：[https://aclanthology.org/2021.sdp-1.4/](https://aclanthology.org/2021.sdp-1.4/)

BDKG at MEDIQA 2021: System Report for the Radiology Report Summarization Task

论文地址：[https://aclanthology.org/2021.bionlp-1.11/](https://aclanthology.org/2021.bionlp-1.11/)

## 3.AAAI 2021

Subtype-Aware Unsupervised Domain Adaptation for Medical Diagnosis

论文地址：[https://arxiv.org/pdf/2101.00318.pdf](https://arxiv.org/pdf/2101.00318.pdf)


Graph-Evolving Meta-Learning for Low-Resource Medical Dialogue Generation

论文地址：[https://arxiv.org/pdf/2012.11988.pdf](https://arxiv.org/pdf/2012.11988.pdf)


A Lightweight Neural Model for Biomedical Entity Linking

论文地址：[https://arxiv.org/pdf/2012.08844.pdf](https://arxiv.org/pdf/2012.08844.pdf)


Automated Lay Language Summarization of Biomedical Scientific Reviews

论文地址：[https://arxiv.org/pdf/2012.12573.pdf](https://arxiv.org/pdf/2012.12573.pdf)


MTAAL: Multi-Task Adversarial Active Learning for Medical Named Entity Recognition and Normalization

论文地址：[https://arxiv.org/pdf/1902.10118.pdf](https://arxiv.org/pdf/1902.10118.pdf)


MELINDA: A Multimodal Dataset for Biomedical Experiment Method Classification

论文地址：[https://arxiv.org/pdf/2012.09216.pdf](https://arxiv.org/pdf/2012.09216.pdf)


## 4.AAAI 2020

Simultaneously Linking Entities and Extracting Relations from Biomedical Text without Mention-Level Supervision

论文地址：[https://aaai.org/ojs/index.php/AAAI/article/view/6236](https://aaai.org/ojs/index.php/AAAI/article/view/6236)


Can Embeddings Adequately Represent Medical Terminology? New Large-Scale Medical Term Similarity Datasets Have the Answer!

论文地址：[https://aaai.org/ojs/index.php/AAAI/article/view/6404](https://aaai.org/ojs/index.php/AAAI/article/view/6404)


Understanding Medical Conversations with Scattered Keyword Attention and Weak Supervision from Responses

论文地址：[https://aaai.org/ojs/index.php/AAAI/article/view/6412](https://aaai.org/ojs/index.php/AAAI/article/view/6412)


Learning Conceptual-Contextual Embeddings for Medical Text

论文地址：[https://aaai.org/ojs/index.php/AAAI/article/view/6504](https://aaai.org/ojs/index.php/AAAI/article/view/6504)


LATTE: Latent Type Modeling for Biomedical Entity Linking

论文地址：[https://aaai.org/ojs/index.php/AAAI/article/view/6526](https://aaai.org/ojs/index.php/AAAI/article/view/6526)



## 5.EMNLP 2021
### 文本分类

**Classification of hierarchical text using geometric deep learning: the case of clinical trials corpus**

使用几何深度学习对分层文本进行分类：以临床试验语料库为例

论文地址：[https://aclanthology.org/2021.emnlp-main.48/](https://aclanthology.org/2021.emnlp-main.48/)

摘要：

We consider the hierarchical representation of documents as graphs and use geometric deep learning to classify them into different categories. While graph neural networks can efficiently handle the variable structure of hierarchical documents using the permutation invariant message passing operations, we show that we can gain extra performance improvements using our proposed selective graph pooling operation that arises from the fact that some parts of the hierarchy are invariable across different documents. We applied our model to classify clinical trial (CT) protocols into completed and terminated categories. We use bag-of-words based, as well as pre-trained transformer-based embeddings to featurize the graph nodes, achieving f1-scoresaround 0.85 on a publicly available large scale CT registry of around 360K protocols. We further demonstrate how the selective pooling can add insights into the CT termination status prediction. We make the source code and dataset splits accessible.



**Effective Convolutional Attention Network for Multi-label Clinical Document Classification**

用于多标签临床文件分类的有效卷积注意网络

论文地址：[https://aclanthology.org/2021.emnlp-main.481/](https://aclanthology.org/2021.emnlp-main.481/)

摘要：

Multi-label document classification (MLDC) problems can be challenging, especially for long documents with a large label set and a long-tail distribution over labels. In this paper, we present an effective convolutional attention network for the MLDC problem with a focus on medical code prediction from clinical documents. Our innovations are three-fold: (1) we utilize a deep convolution-based encoder with the squeeze-and-excitation networks and residual networks to aggregate the information across the document and learn meaningful document representations that cover different ranges of texts; (2) we explore multi-layer and sum-pooling attention to extract the most informative features from these multi-scale representations; (3) we combine binary cross entropy loss and focal loss to improve performance for rare labels. We focus our evaluation study on MIMIC-III, a widely used dataset in the medical domain. Our models outperform prior work on medical coding and achieve new state-of-the-art results on multiple metrics. We also demonstrate the language independent nature of our approach by applying it to two non-English datasets. Our model outperforms prior best model and a multilingual Transformer model by a substantial margin.



**[Description-based Label Attention Classifier for Explainable ICD-9 Classification]()**

用于可解释 ICD-9 分类的基于描述的标签注意分类器

论文地址：[https://aclanthology.org/2021.wnut-1.8/](https://aclanthology.org/2021.wnut-1.8/)

摘要：

ICD-9 coding is a relevant clinical billing task, where unstructured texts with information about a patient’s diagnosis and treatments are annotated with multiple ICD-9 codes. Automated ICD-9 coding is an active research field, where CNN- and RNN-based model architectures represent the state-of-the-art approaches. In this work, we propose a description-based label attention classifier to improve the model explainability when dealing with noisy texts like clinical notes.

### 关系抽取

**Incorporating medical knowledge in BERT for clinical relation extraction**

在 BERT 中结合医学知识进行临床关系提取

论文地址：[https://aclanthology.org/2021.emnlp-main.435/](https://aclanthology.org/2021.emnlp-main.435/)

摘要：

In recent years pre-trained language models (PLM) such as BERT have proven to be very effective in diverse NLP tasks such as Information Extraction, Sentiment Analysis and Question Answering. Trained with massive general-domain text, these pre-trained language models capture rich syntactic, semantic and discourse information in the text. However, due to the differences between general and specific domain text (e.g., Wikipedia versus clinic notes), these models may not be ideal for domain-specific tasks (e.g., extracting clinical relations). Furthermore, it may require additional medical knowledge to understand clinical text properly. To solve these issues, in this research, we conduct a comprehensive examination of different techniques to add medical knowledge into a pre-trained BERT model for clinical relation extraction. Our best model outperforms the state-of-the-art systems on the benchmark i2b2/VA 2010 clinical relation extraction dataset.



### 实体链接

**BERT might be Overkill: A Tiny but Effective Biomedical Entity Linker based on Residual Convolutional Neural Networks**

BERT 可能有点矫枉过正：基于残差卷积神经网络的微小但有效的生物医学实体链接器

论文地址：[https://aclanthology.org/2021.findings-emnlp.140/](https://aclanthology.org/2021.findings-emnlp.140/)

摘要：

Biomedical entity linking is the task of linking entity mentions in a biomedical document to referent entities in a knowledge base. Recently, many BERT-based models have been introduced for the task. While these models achieve competitive results on many datasets, they are computationally expensive and contain about 110M parameters. Little is known about the factors contributing to their impressive performance and whether the over-parameterization is needed. In this work, we shed some light on the inner workings of these large BERT-based models. Through a set of probing experiments, we have found that the entity linking performance only changes slightly when the input word order is shuffled or when the attention scope is limited to a fixed window size. From these observations, we propose an efficient convolutional neural network with residual connections for biomedical entity linking. Because of the sparse connectivity and weight sharing properties, our model has a small number of parameters and is highly efficient. On five public datasets, our model achieves comparable or even better linking accuracy than the state-of-the-art BERT-based models while having about 60 times fewer parameters.



### 实体消歧

**Coreference Resolution for the Biomedical Domain: A Survey**

生物医学领域的共指解析：一项调查

论文地址：[https://aclanthology.org/2021.crac-1.2/](https://aclanthology.org/2021.crac-1.2/)

摘要：

Issues with coreference resolution are one of the most frequently mentioned challenges for information extraction from the biomedical literature. Thus, the biomedical genre has long been the second most researched genre for coreference resolution after the news domain, and the subject of a great deal of research for NLP in general. In recent years this interest has grown enormously leading to the development of a number of substantial datasets, of domain-specific contextual language models, and of several architectures. In this paper we review the state of-the-art of coreference in the biomedical domain with a particular attention on these most recent developments.



**Cross-Domain Data Integration for Named Entity Disambiguation in Biomedical Text**

生物医学文本中命名实体消歧的跨域数据集成

论文地址：[https://aclanthology.org/2021.findings-emnlp.388/](https://aclanthology.org/2021.findings-emnlp.388/)

摘要：

Named entity disambiguation (NED), which involves mapping textual mentions to structured entities, is particularly challenging in the medical domain due to the presence of rare entities. Existing approaches are limited by the presence of coarse-grained structural resources in biomedical knowledge bases as well as the use of training datasets that provide low coverage over uncommon resources. In this work, we address these issues by proposing a cross-domain data integration method that transfers structural knowledge from a general text knowledge base to the medical domain. We utilize our integration scheme to augment structural resources and generate a large biomedical NED dataset for pretraining. Our pretrained model with injected structural knowledge achieves state-of-the-art performance on two benchmark medical NED datasets: MedMentions and BC5CDR. Furthermore, we improve disambiguation of rare entities by up to 57 accuracy points.



### 医疗概念标准化

**Biomedical Concept Normalization by Leveraging Hypernyms**

利用上位词进行生物医学概念规范化

论文地址：[https://aclanthology.org/2021.emnlp-main.284/](https://aclanthology.org/2021.emnlp-main.284/)

摘要：

Biomedical Concept Normalization (BCN) is widely used in biomedical text processing as a fundamental module. Owing to numerous surface variants of biomedical concepts, BCN still remains challenging and unsolved. In this paper, we exploit biomedical concept hypernyms to facilitate BCN. We propose Biomedical Concept Normalizer with Hypernyms (BCNH), a novel framework that adopts list-wise training to make use of both hypernyms and synonyms, and also employs norm constraint on the representation of hypernym-hyponym entity pairs. The experimental results show that BCNH outperforms the previous state-of-the-art model on the NCBI dataset.



### 医学知识图谱

**Mixture-of-Partitions: Infusing Large Biomedical Knowledge Graphs into BERT**

分区混合：将大型生物医学知识图注入 BERT

论文地址：[https://aclanthology.org/2021.emnlp-main.383/](https://aclanthology.org/2021.emnlp-main.383/)

摘要：

Infusing factual knowledge into pre-trained models is fundamental for many knowledge-intensive tasks. In this paper, we proposed Mixture-of-Partitions (MoP), an infusion approach that can handle a very large knowledge graph (KG) by partitioning it into smaller sub-graphs and infusing their specific knowledge into various BERT models using lightweight adapters. To leverage the overall factual knowledge for a target task, these sub-graph adapters are further fine-tuned along with the underlying BERT through a mixture layer. We evaluate our MoP with three biomedical BERTs (SciBERT, BioBERT, PubmedBERT) on six downstream tasks (inc. NLI, QA, Classification), and the results show that our MoP consistently enhances the underlying BERTs in task performance, and achieves new SOTA performances on five evaluated datasets.



**Can Language Models be Biomedical Knowledge Bases?**

语言模型可以成为生物医学知识库吗？

论文地址：[https://aclanthology.org/2021.emnlp-main.388/](https://aclanthology.org/2021.emnlp-main.388/)

摘要：

Pre-trained language models (LMs) have become ubiquitous in solving various natural language processing (NLP) tasks. There has been increasing interest in what knowledge these LMs contain and how we can extract that knowledge, treating LMs as knowledge bases (KBs). While there has been much work on probing LMs in the general domain, there has been little attention to whether these powerful LMs can be used as domain-specific KBs. To this end, we create the BioLAMA benchmark, which is comprised of 49K biomedical factual knowledge triples for probing biomedical LMs. We find that biomedical LMs with recently proposed probing methods can achieve up to 18.51% Acc@5 on retrieving biomedical knowledge. Although this seems promising given the task difficulty, our detailed analyses reveal that most predictions are highly correlated with prompt templates without any subjects, hence producing similar results on each relation and hindering their capabilities to be used as domain-specific KBs. We hope that BioLAMA can serve as a challenging benchmark for biomedical factual probing.



### QA

**MLEC-QA: A Chinese Multi-Choice Biomedical Question Answering Dataset**

MLEC-QA：中文多选生物医学问答数据集

论文地址：[https://aclanthology.org/2021.emnlp-main.698/](https://aclanthology.org/2021.emnlp-main.698/)

Question Answering (QA) has been successfully applied in scenarios of human-computer interaction such as chatbots and search engines. However, for the specific biomedical domain, QA systems are still immature due to expert-annotated datasets being limited by category and scale. In this paper, we present MLEC-QA, the largest-scale Chinese multi-choice biomedical QA dataset, collected from the National Medical Licensing Examination in China. The dataset is composed of five subsets with 136,236 biomedical multi-choice questions with extra materials (images or tables) annotated by human experts, and first covers the following biomedical sub-fields: Clinic, Stomatology, Public Health, Traditional Chinese Medicine, and Traditional Chinese Medicine Combined with Western Medicine. We implement eight representative control methods and open-domain QA methods as baselines. Experimental results demonstrate that even the current best model can only achieve accuracies between 40% to 55% on five subsets, especially performing poorly on questions that require sophisticated reasoning ability. We hope the release of the MLEC-QA dataset can serve as a valuable resource for research and evaluation in open-domain QA, and also make advances for biomedical QA systems.



**What Would it Take to get Biomedical QA Systems into Practice?**

将生物医学 QA 系统付诸实践需要什么？

论文地址：[https://aclanthology.org/2021.mrqa-1.3/](https://aclanthology.org/2021.mrqa-1.3/)

摘要：

Medical question answering (QA) systems have the potential to answer clinicians’ uncertainties about treatment and diagnosis on-demand, informed by the latest evidence. However, despite the significant progress in general QA made by the NLP community, medical QA systems are still not widely used in clinical environments. One likely reason for this is that clinicians may not readily trust QA system outputs, in part because transparency, trustworthiness, and provenance have not been key considerations in the design of such models. In this paper we discuss a set of criteria that, if met, we argue would likely increase the utility of biomedical QA systems, which may in turn lead to adoption of such systems in practice. We assess existing models, tasks, and datasets with respect to these criteria, highlighting shortcomings of previously proposed approaches and pointing toward what might be more usable QA systems.



### 文本生成

**Automated Generation of Accurate & Fluent Medical X-ray Reports**

自动生成准确流畅的医学 X 射线报告

论文地址：[https://aclanthology.org/2021.emnlp-main.288/](https://aclanthology.org/2021.emnlp-main.288/)

摘要：

Our paper aims to automate the generation of medical reports from chest X-ray image inputs, a critical yet time-consuming task for radiologists. Existing medical report generation efforts emphasize producing human-readable reports, yet the generated text may not be well aligned to the clinical facts. Our generated medical reports, on the other hand, are fluent and, more importantly, clinically accurate. This is achieved by our fully differentiable and end-to-end paradigm that contains three complementary modules: taking the chest X-ray images and clinical history document of patients as inputs, our classification module produces an internal checklist of disease-related topics, referred to as enriched disease embedding; the embedding representation is then passed to our transformer-based generator, to produce the medical report; meanwhile, our generator also creates a weighted embedding representation, which is fed to our interpreter to ensure consistency with respect to disease-related topics. Empirical evaluations demonstrate very promising results achieved by our approach on commonly-used metrics concerning language fluency and clinical accuracy. Moreover, noticeable performance gains are consistently observed when additional input information is available, such as the clinical document and extra scans from different views.



### 其他

**Zero-Shot Clinical Questionnaire Filling From Human-Machine Interactions**

从人机交互中填写零样本临床问卷

论文地址：[https://aclanthology.org/2021.mrqa-1.5/](https://aclanthology.org/2021.mrqa-1.5/)

摘要：

In clinical studies, chatbots mimicking doctor-patient interactions are used for collecting information about the patient’s health state. Later, this information needs to be processed and structured for the doctor. One way to organize it is by automatically filling the questionnaires from the human-bot conversation. It would help the doctor to spot the possible issues. Since there is no such dataset available for this task and its collection is costly and sensitive, we explore the capacities of state-of-the-art zero-shot models for question answering, textual inference, and text classification. We provide a detailed analysis of the results and propose further directions for clinical questionnaire filling.



**Robustness and Sensitivity of BERT Models Predicting Alzheimer’s Disease from Text**

论文地址：[https://aclanthology.org/2021.wnut-1.37/](https://aclanthology.org/2021.wnut-1.37/)

论文地址：

摘要：

Understanding robustness and sensitivity of BERT models predicting Alzheimer’s disease from text is important for both developing better classification models and for understanding their capabilities and limitations. In this paper, we analyze how a controlled amount of desired and undesired text alterations impacts performance of BERT. We show that BERT is robust to natural linguistic variations in text. On the other hand, we show that BERT is not sensitive to removing clinically important information from text.



**Interacting Knowledge Sources, Inspection and Analysis: Case-studies on Biomedical text processing**

交互知识源、检查和分析：生物医学文本处理案例研究

论文地址：[https://aclanthology.org/2021.blackboxnlp-1.35/](https://aclanthology.org/2021.blackboxnlp-1.35/)

摘要：

In this paper we investigate the recently proposed multi-input RIM for inspectability. This framework follows an encapsulation paradigm, where external knowledge sources are encoded as largely independent modules, enabling transparency for model inspection.





**Sent2Span: Span Detection for PICO Extraction in the Biomedical Text without Span Annotations**

Sent2Span：无跨度注释的生物医学文本中 PICO 提取的跨度检测

论文地址：[https://aclanthology.org/2021.findings-emnlp.147/](https://aclanthology.org/2021.findings-emnlp.147/)

摘要：

The rapid growth in published clinical trials makes it difficult to maintain up-to-date systematic reviews, which require finding all relevant trials. This leads to policy and practice decisions based on out-of-date, incomplete, and biased subsets of available clinical evidence. Extracting and then normalising Population, Intervention, Comparator, and Outcome (PICO) information from clinical trial articles may be an effective way to automatically assign trials to systematic reviews and avoid searching and screening—the two most time-consuming systematic review processes. We propose and test a novel approach to PICO span detection. The major difference between our proposed method and previous approaches comes from detecting spans without needing annotated span data and using only crowdsourced sentence-level annotations. Experiments on two datasets show that PICO span detection results achieve much higher results for recall when compared to fully supervised methods with PICO sentence detection at least as good as human annotations. By removing the reliance on expert annotations for span detection, this work could be used in a human-machine pipeline for turning low-quality, crowdsourced, and sentence-level PICO annotations into structured information that can be used to quickly assign trials to relevant systematic reviews.



**Parameter-Efficient Domain Knowledge Integration from Multiple Sources for Biomedical Pre-trained Language Models**

用于生物医学预训练语言模型的多源参数高效领域知识集成

论文地址：[https://aclanthology.org/2021.findings-emnlp.325/](https://aclanthology.org/2021.findings-emnlp.325/)

摘要：

Domain-specific pre-trained language models (PLMs) have achieved great success over various downstream tasks in different domains. However, existing domain-specific PLMs mostly rely on self-supervised learning over large amounts of domain text, without explicitly integrating domain-specific knowledge, which can be essential in many domains. Moreover, in knowledge-sensitive areas such as the biomedical domain, knowledge is stored in multiple sources and formats, and existing biomedical PLMs either neglect them or utilize them in a limited manner. In this work, we introduce an architecture to integrate domain knowledge from diverse sources into PLMs in a parameter-efficient way. More specifically, we propose to encode domain knowledge via *adapters*, which are small bottleneck feed-forward networks inserted between intermediate transformer layers in PLMs. These knowledge adapters are pre-trained for individual domain knowledge sources and integrated via an attention-based knowledge controller to enrich PLMs. Taking the biomedical domain as a case study, we explore three knowledge-specific adapters for PLMs based on the UMLS Metathesaurus graph, the Wikipedia articles for diseases, and the semantic grouping information for biomedical concepts. Extensive experiments on different biomedical NLP tasks and datasets demonstrate the benefits of the proposed architecture and the knowledge-specific adapters across multiple PLMs.





**MSˆ2: Multi-Document Summarization of Medical Studies**

MSˆ2：医学研究的多文档摘要

论文地址：[https://aclanthology.org/2021.emnlp-main.594/](https://aclanthology.org/2021.emnlp-main.594/)

摘要：

To assess the effectiveness of any medical intervention, researchers must conduct a time-intensive and manual literature review. NLP systems can help to automate or assist in parts of this expensive process. In support of this goal, we release MSˆ2 (Multi-Document Summarization of Medical Studies), a dataset of over 470k documents and 20K summaries derived from the scientific literature. This dataset facilitates the development of systems that can assess and aggregate contradictory evidence across multiple studies, and is the first large-scale, publicly available multi-document summarization dataset in the biomedical domain. We experiment with a summarization system based on BART, with promising early results, though significant work remains to achieve higher summarization quality. We formulate our summarization inputs and targets in both free text and structured forms and modify a recently proposed metric to assess the quality of our system’s generated summaries. Data and models are available at https://github.com/allenai/ms2.





**Detecting Health Advice in Medical Research Literature**

检测医学研究文献中的健康建议

论文地址：[https://aclanthology.org/2021.emnlp-main.486/](https://aclanthology.org/2021.emnlp-main.486/)

摘要：

Health and medical researchers often give clinical and policy recommendations to inform health practice and public health policy. However, no current health information system supports the direct retrieval of health advice. This study fills the gap by developing and validating an NLP-based prediction model for identifying health advice in research publications. We annotated a corpus of 6,000 sentences extracted from structured abstracts in PubMed publications as ‘“strong advice”, “weak advice”, or “no advice”, and developed a BERT-based model that can predict, with a macro-averaged F1-score of 0.93, whether a sentence gives strong advice, weak advice, or not. The prediction model generalized well to sentences in both unstructured abstracts and discussion sections, where health advice normally appears. We also conducted a case study that applied this prediction model to retrieve specific health advice on COVID-19 treatments from LitCovid, a large COVID research literature portal, demonstrating the usefulness of retrieving health advice sentences as an advanced research literature navigation function for health researchers and the general public.





[

](https://aclanthology.org/attachments/2021.findings-emnlp.300.Software.zip)**Exploring a Unified Sequence-To-Sequence Transformer for Medical Product Safety Monitoring in Social Media**

探索用于社交媒体中医疗产品安全监控的统一序列到序列转换器

论文地址：[https://aclanthology.org/2021.findings-emnlp.300/](https://aclanthology.org/2021.findings-emnlp.300/)

摘要：

Adverse Events (AE) are harmful events resulting from the use of medical products. Although social media may be crucial for early AE detection, the sheer scale of this data makes it logistically intractable to analyze using human agents, with NLP representing the only low-cost and scalable alternative. In this paper, we frame AE Detection and Extraction as a sequence-to-sequence problem using the T5 model architecture and achieve strong performance improvements over the baselines on several English benchmarks (F1 = 0.71, 12.7% relative improvement for AE Detection; Strict F1 = 0.713, 12.4% relative improvement for AE Extraction). Motivated by the strong commonalities between AE tasks, the class imbalance in AE benchmarks, and the linguistic and structural variety typical of social media texts, we propose a new strategy for multi-task training that accounts, at the same time, for task and dataset characteristics. Our approach increases model robustness, leading to further performance gains. Finally, our framework shows some language transfer capabilities, obtaining higher performance than Multilingual BERT in zero-shot learning on French data.





**[How to leverage the multimodal EHR data for better medical prediction?**

如何利用多模态 EHR 数据进行更好的医学预测

论文地址：[https://aclanthology.org/2021.emnlp-main.329/](https://aclanthology.org/2021.emnlp-main.329/)

摘要：

Healthcare is becoming a more and more important research topic recently. With the growing data in the healthcare domain, it offers a great opportunity for deep learning to improve the quality of service and reduce costs. However, the complexity of electronic health records (EHR) data is a challenge for the application of deep learning. Specifically, the data produced in the hospital admissions are monitored by the EHR system, which includes structured data like daily body temperature and unstructured data like free text and laboratory measurements. Although there are some preprocessing frameworks proposed for specific EHR data, the clinical notes that contain significant clinical value are beyond the realm of their consideration. Besides, whether these different data from various views are all beneficial to the medical tasks and how to best utilize these data remain unclear. Therefore, in this paper, we first extract the accompanying clinical notes from EHR and propose a method to integrate these data, we also comprehensively study the different models and the data leverage methods for better medical task prediction performance. The results on two prediction tasks show that our fused model with different data outperforms the state-of-the-art method without clinical notes, which illustrates the importance of our fusion method and the clinical note features.



**Leveraging Capsule Routing to Associate Knowledge with Medical Literature Hierarchically**

利用胶囊路由将知识与医学文献分层关联

论文地址：[https://aclanthology.org/2021.emnlp-main.285/](https://aclanthology.org/2021.emnlp-main.285/)

摘要：

Integrating knowledge into text is a promising way to enrich text representation, especially in the medical field. However, undifferentiated knowledge not only confuses the text representation but also imports unexpected noises. In this paper, to alleviate this problem, we propose leveraging capsule routing to associate knowledge with medical literature hierarchically (called HiCapsRKL). Firstly, HiCapsRKL extracts two empirically designed text fragments from medical literature and encodes them into fragment representations respectively. Secondly, the capsule routing algorithm is applied to two fragment representations. Through the capsule computing and dynamic routing, each representation is processed into a new representation (denoted as caps-representation), and we integrate the caps-representations as information gain to associate knowledge with medical literature hierarchically. Finally, HiCapsRKL are validated on relevance prediction and medical literature retrieval test sets. The experimental results and analyses show that HiCapsRKLcan more accurately associate knowledge with medical literature than mainstream methods. In summary, HiCapsRKL can efficiently help selecting the most relevant knowledge to the medical literature, which may be an alternative attempt to improve knowledge-based text representation. Source code is released on GitHub.




## 6.EMNLP 2020

Infusing Disease Knowledge into BERT for Health Question Answering, Medical Inference and Disease Name Recognition
 
论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.372/](https://www.aclweb.org/anthology/2020.emnlp-main.372/)


### 机器翻译

Evaluation of Machine Translation Methods applied to Medical Terminologies

论文地址：[https://www.aclweb.org/anthology/2020.louhi-1.7/](https://www.aclweb.org/anthology/2020.louhi-1.7/)


A Multilingual Neural Machine Translation Model for Biomedical Data

论文地址：[https://www.aclweb.org/anthology/2020.nlpcovid19-2.16/](https://www.aclweb.org/anthology/2020.nlpcovid19-2.16/)


Findings of the WMT 2020 Biomedical Translation Shared Task: Basque, Italian and Russian as New Additional Languages

论文地址：[https://www.aclweb.org/anthology/2020.wmt-1.76/](https://www.aclweb.org/anthology/2020.wmt-1.76/)


Elhuyar submission to the Biomedical Translation Task 2020 on terminology and abstracts translation

论文地址：[https://www.aclweb.org/anthology/2020.wmt-1.87/](https://www.aclweb.org/anthology/2020.wmt-1.87/)


Pretrained Language Models and Backtranslation for English-Basque Biomedical Neural Machine Translation

论文地址：[https://www.aclweb.org/anthology/2020.wmt-1.89/](https://www.aclweb.org/anthology/2020.wmt-1.89/)



### 机器阅读理解

Towards Medical Machine Reading Comprehension with Structural Knowledge and Plain Text

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.111/](https://www.aclweb.org/anthology/2020.emnlp-main.111/)



### 实体规范化

A Knowledge-driven Generative Model for Multi-implication Chinese Medical Procedure Entity Normalization

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.116/](https://www.aclweb.org/anthology/2020.emnlp-main.116/)


Target Concept Guided Medical Concept Normalization in Noisy User-Generated Texts

论文地址：[https://www.aclweb.org/anthology/2020.deelio-1.8/](https://www.aclweb.org/anthology/2020.deelio-1.8/)


Medical Concept Normalization in User-Generated Texts by Learning Target Concept Embeddings

论文地址：[https://www.aclweb.org/anthology/2020.louhi-1.3/](https://www.aclweb.org/anthology/2020.louhi-1.3/)



### 命名实体识别

Assessment of DistilBERT performance on Named Entity Recognition task for the detection of Protected Health Information and medical concepts

论文地址：[https://www.aclweb.org/anthology/2020.clinicalnlp-1.18/](https://www.aclweb.org/anthology/2020.clinicalnlp-1.18/)



### 关系抽取

FedED: Federated Learning via Ensemble Distillation for Medical Relation Extraction

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.165/](https://www.aclweb.org/anthology/2020.emnlp-main.165/)



### 实体链接

COMETA: A Corpus for Medical Entity Linking in the Social Media

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.253/](https://www.aclweb.org/anthology/2020.emnlp-main.253/)



Simple Hierarchical Multi-Task Neural End-To-End Entity Linking for Biomedical Text

论文地址：[https://www.aclweb.org/anthology/2020.louhi-1.2/](https://www.aclweb.org/anthology/2020.louhi-1.2/)



### 语言模型

BioMegatron: Larger Biomedical Domain Language Model

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.379/](https://www.aclweb.org/anthology/2020.emnlp-main.379/)


Pretrained Language Models for Biomedical and Clinical Tasks: Understanding and Extending the State-of-the-Art

论文地址：[https://www.aclweb.org/anthology/2020.clinicalnlp-1.17/](https://www.aclweb.org/anthology/2020.clinicalnlp-1.17/)


Inexpensive Domain Adaptation of Pretrained Language Models: Case Studies on Biomedical NER and Covid-19 QA

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.134/](https://www.aclweb.org/anthology/2020.findings-emnlp.134/)


On the effectiveness of small, discriminatively pre-trained language representation models for biomedical text mining

论文地址：[https://www.aclweb.org/anthology/2020.sdp-1.12/](https://www.aclweb.org/anthology/2020.sdp-1.12/)



### 事件抽取

Biomedical Event Extraction as Sequence Labeling

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.431/](https://www.aclweb.org/anthology/2020.emnlp-main.431/)


Biomedical Event Extraction with Hierarchical Knowledge Graphs

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.114/](https://www.aclweb.org/anthology/2020.findings-emnlp.114/)



### 数据集

COMETA: A Corpus for Medical Entity Linking in the Social Media

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.253/](https://www.aclweb.org/anthology/2020.emnlp-main.253/)


MedDialog: Large-scale Medical Dialogue Datasets

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.743/](https://www.aclweb.org/anthology/2020.emnlp-main.743/)


MeDAL: Medical Abbreviation Disambiguation Dataset for Natural Language Understanding Pretraining

论文地址：[https://www.aclweb.org/anthology/2020.clinicalnlp-1.15/](https://www.aclweb.org/anthology/2020.clinicalnlp-1.15/)


MedICaT: A Dataset of Medical Images, Captions, and Textual References

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.191/](https://www.aclweb.org/anthology/2020.findings-emnlp.191/)


GGPONC: A Corpus of German Medical Text with Rich Metadata Based on Clinical Practice Guidelines

论文地址：[https://www.aclweb.org/anthology/2020.louhi-1.5/](https://www.aclweb.org/anthology/2020.louhi-1.5/)



### 基于国外临床医学数据的NLP研究

Information Extraction from Swedish Medical Prescriptions with Sig-Transformer Encoder

论文地址：[https://www.aclweb.org/anthology/2020.clinicalnlp-1.5/](https://www.aclweb.org/anthology/2020.clinicalnlp-1.5/)


Classification of Syncope Cases in Norwegian Medical Records

论文地址：[https://www.aclweb.org/anthology/2020.clinicalnlp-1.9/](https://www.aclweb.org/anthology/2020.clinicalnlp-1.9/)



### 对话

Weakly Supervised Medication Regimen Extraction from Medical Conversations

论文地址：[https://www.aclweb.org/anthology/2020.clinicalnlp-1.20/](https://www.aclweb.org/anthology/2020.clinicalnlp-1.20/)


Dr. Summarize: Global Summarization of Medical Dialogue by Exploiting Local Structures.

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.335/](https://www.aclweb.org/anthology/2020.findings-emnlp.335/)



### 文本生成

Reinforcement Learning with Imbalanced Dataset for Data-to-Text Medical Report Generation

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.202/](https://www.aclweb.org/anthology/2020.findings-emnlp.202/)


Generating Accurate Electronic Health Assessment from Medical Graph

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.336/](https://www.aclweb.org/anthology/2020.findings-emnlp.336/)



### 问答

Biomedical Event Extraction as Multi-turn Question Answering

论文地址：[https://www.aclweb.org/anthology/2020.louhi-1.10/](https://www.aclweb.org/anthology/2020.louhi-1.10/)



### 推荐

COVID-19: A Semantic-Based Pipeline for Recommending Biomedical Entities

论文地址：[https://www.aclweb.org/anthology/2020.nlpcovid19-2.20/](https://www.aclweb.org/anthology/2020.nlpcovid19-2.20/)



### 主题模型


Developing a Curated Topic Model for COVID-19 Medical Research Literature

论文地址：[https://www.aclweb.org/anthology/2020.nlpcovid19-2.30/](https://www.aclweb.org/anthology/2020.nlpcovid19-2.30/)



### 表示学习

ERLKG: Entity Representation Learning and Knowledge Graph based association analysis of COVID-19 through mining of unstructured biomedical corpora

论文地址：[https://www.aclweb.org/anthology/2020.sdp-1.15/](https://www.aclweb.org/anthology/2020.sdp-1.15/)


Learning Informative Representations of Biomedical Relations with Latent Variable Models

论文地址：[https://www.aclweb.org/anthology/2020.sustainlp-1.3/](https://www.aclweb.org/anthology/2020.sustainlp-1.3/)



### Others

Dilated Convolutional Attention Network for Medical Code Assignment from Clinical Text

论文地址：[https://www.aclweb.org/anthology/2020.clinicalnlp-1.8/](https://www.aclweb.org/anthology/2020.clinicalnlp-1.8/)


Summarizing Chinese Medical Answer with Graph Convolution Networks and Question-focused Dual Attention

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.2/](https://www.aclweb.org/anthology/2020.findings-emnlp.2/)


Sequential Span Classification with Neural Semi-Markov CRFs for Biomedical Abstracts

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.77/](https://www.aclweb.org/anthology/2020.findings-emnlp.77/)


Characterizing the Value of Information in Medical Notes

论文地址：[https://www.aclweb.org/anthology/2020.findings-emnlp.187/](https://www.aclweb.org/anthology/2020.findings-emnlp.187/)


Querying Across Genres for Medical Claims in News

论文地址：[https://www.aclweb.org/anthology/2020.emnlp-main.139/](https://www.aclweb.org/anthology/2020.emnlp-main.139/)


An efficient representation of chronological events in medical texts

论文地址：[https://www.aclweb.org/anthology/2020.louhi-1.11/](
