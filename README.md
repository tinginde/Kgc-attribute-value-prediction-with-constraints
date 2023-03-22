<<<<<<< HEAD
Project Title: Numerical Attribute Prediction in Knowledge Graphs with Constraints
Introduction
Knowledge Graph Completion is an important task for machines to utilize structural information provided by knowledge graphs to improve performance in various applications. However, current approaches to knowledge graph completion ignore the existing numerical data in the graphs, leading to a lack of understanding of the data and poor predicted performance. In this project, we aim to predict missing numerical attribute values for entities in KGs by incorporating arithmetic relationships between attributes as constraints in the models.

Methodology
To achieve this goal, we first defined three types of constraint shapes and identified seven constraint rules from a benchmark dataset called Literally-Wikidata. We then built a regression model and a multi-task approach model and incorporated constraints into the model through two experiments: a single-attribute regression model that added relevant attribute values as features, and a single-attribute prediction model that incorporated constraints into the loss function.

Results
Our results show that incorporating constraints can guide the model to make more accurate predictions for some attributes such as "height", but overall, it does not significantly improve the performance of the model. Nevertheless, this study provides a novel approach to incorporating arithmetic relationships between attributes in KG completion tasks, and can potentially be extended to other types of constraints to improve the accuracy of predictions.
=======
Numerical Attribute Prediction in Knowledge Graphs with Constraints
== 
This repository consists of:
* Preprocessed Numercial Triples in LiterallyWikidata
* Models to predict numeric values of attributes in knowledge graphs:
  * Regression model
  * MT-KGNN model (revised from Dadoun, 2021)

Introduction
--
Knowledge Graph Completion is an important task for machines to utilize structural information provided by knowledge graphs to improve performance in various applications. However, current approaches to knowledge graph completion ignore the existing numerical data in the graphs, leading to a lack of understanding of the data and poor predicted performance. In this project, we aim to predict missing numerical attribute values for entities in KGs by incorporating arithmetic relationships between attributes as constraints in the models.

Methodology
--
To achieve this goal, we first defined three types of constraint shapes and identified seven constraint rules from a benchmark dataset called LiterallyWikidata (Gesese et al., 2021a). We then built a regression model and a multi-task approach model and incorporated constraints into the model through two experiments: a single-attribute regression model that added relevant attribute values as features, and a single-attribute prediction model that incorporated constraints into the loss function.

Results
--
Our results show that incorporating constraints can guide the model to make more accurate predictions for some attributes such as "height", but overall, it does not significantly improve the performance of the model. Nevertheless, this study provides a novel approach to incorporating arithmetic relationships between attributes in KG completion tasks, and can potentially be extended to other types of constraints to improve the accuracy of predictions.

References
--
1. Amine Dadoun, Raphael Troncy, Michael Defoin-Platel, and Gerardo Ayala Solano.
Predicting your next trip: A knowledge graph-based multi-task learning approach
for travel destination recommendation. 2021 Workshop on Recommenders in
Tourism, RecTour 2021, 2974, 2021. Gitlab: https://gitlab.eurecom.fr/amadeus/KGMTL4Rec
2. Genet Asefa Gesese, Mehwish Alam, and Harald Sack. Literallywikidata - a benchmark
for knowledge graph completion using literals. In SEMWEB, 2021.
>>>>>>> 69b26c8aee92f10f3de85a5b4fc317ba5b92ec2f
