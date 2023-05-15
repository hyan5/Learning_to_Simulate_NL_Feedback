# Learning to Simulate Natural Language Feedback for Interactive Semantic Parsing
This repository provides code implementation for our paper [Learning to Simulate Natural Language Feedback for Interactive Semantic Parsing](https://github.com/hyan5/Learning_to_Simulate_NL_Feedback.git) accepted by *ACL 2023*.

## 1. Overview
Interactive semantic parsing based on natural language (NL) feedback, where users provide feedback to correct the parser mistakes, has emerged as a more practical scenario than the traditional one-shot semantic parsing. However, prior work has heavily relied on human- annotated feedback data to train the interactive semantic parser, which is prohibitively expensive and not scalable. In this work, we pro- pose a new task of simulating NL feedback for interactive semantic parsing. We accompany the task with a novel feedback evaluator. The evaluator is specifically designed to assess the quality of the simulated feedback, based on which we decide the best feedback simulator from our proposed variants. On a text-to-SQL dataset, we show that our feedback simulator can generate high-quality NL feedback to boost the error correction ability of a specific parser. In low-data settings, our feedback simulator can help achieve comparable error correction performance as trained using the costly, full set of human annotations.
<p align="center">
<img src="overview.png" alt="Arch Overview" title="Overview" width="600"/>
</p>
