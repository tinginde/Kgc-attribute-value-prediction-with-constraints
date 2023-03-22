Project Title: Numerical Attribute Prediction in Knowledge Graphs with Constraints
==
Introduction
--
Knowledge Graph Completion is an important task for machines to utilize structural information provided by knowledge graphs to improve performance in various applications. However, current approaches to knowledge graph completion ignore the existing numerical data in the graphs, leading to a lack of understanding of the data and poor predicted performance. In this project, we aim to predict missing numerical attribute values for entities in KGs by incorporating arithmetic relationships between attributes as constraints in the models.

Methodology
--
To achieve this goal, we first defined three types of constraint shapes and identified seven constraint rules from a benchmark dataset called LiterallyWikidata (Gesese et al., 2021a). We then built a regression model and a multi-task approach model and incorporated constraints into the model through two experiments: a single-attribute regression model that added relevant attribute values as features, and a single-attribute prediction model that incorporated constraints into the loss function.

Results
--
Our results show that incorporating constraints can guide the model to make more accurate predictions for some attributes such as "height", but overall, it does not significantly improve the performance of the model. Nevertheless, this study provides a novel approach to incorporating arithmetic relationships between attributes in KG completion tasks, and can potentially be extended to other types of constraints to improve the accuracy of predictions.
