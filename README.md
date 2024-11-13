# K-Means Clustering of Stock Data

This project applies K-means clustering, with dimensionality reduction via t-SNE, to identify distinct groups of companies in the stock market (S&P 500 from 2010 to 2020) based on their performance and sector information. The analysis explores how clustering can reveal patterns across different industries, highlight key performance attributes, and offer insights into company relationships within and between sectors.

## Project Overview

The purpose of this project is to categorize stocks based on their performance characteristics using K-means clustering. By leveraging sector information and clustering results, we gain insights into commonalities and differences among companies across sectors and stages of production.

This analysis highlights both well-defined clusters (e.g., financial services, healthcare) and boundary clusters that may represent cross-sector entities or require further fine-tuning.

## Project Objectives

- **Data Collection**: Gather historical stock price data for S&P 500 companies using the Yahoo Finance API.
- **Preprocessing**: Standardize the data to ensure consistent scaling, essential for clustering algorithms.
- **Clustering Analysis**: Use machine learning techniques like K-Means to cluster companies based on their historical stock price data.
- **Dimensionality Reduction and Visualization**: Apply techniques such as PCA (Principal Component Analysis) and t-SNE (t-distributed Stochastic Neighbor Embedding) to reduce the complexity of the data, allowing for meaningful visualizations of the clustered groups.
