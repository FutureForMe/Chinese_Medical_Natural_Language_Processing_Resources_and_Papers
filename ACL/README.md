# Medical NLP Papers in ACL
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
