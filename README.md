# Movie Recommendation System

A comprehensive machine learning project implementing multiple collaborative filtering and content-based recommendation algorithms using Amazon Movies & TV reviews dataset.

## 📋 Overview

This project explores various recommendation approaches from basic popularity-based methods to advanced collaborative filtering techniques. It demonstrates data preprocessing, baseline models, similarity metrics, and evaluation strategies for building production-ready recommendation systems.

## 🎯 Key Features

### Data Processing
- **K-core filtering**: Ensures minimum rating density and stability
- **Amazon Reviews dataset**: 500K+ reviews from Movies & TV category  
- **Metadata enrichment**: Item titles and metadata integration
- **Data cleaning**: Duplicate removal and missing value handling

### Recommendation Algorithms

1. **Baseline Models**
   - Global Mean Rating
   - Item Mean Rating
   - Top-K Popularity

2. **Collaborative Filtering**
   - User-User Similarity (Pearson Correlation)
   - Item-Item Similarity
   - Hybrid approaches

3. **Matrix Factorization** (via Surprise library)
   - SVD (Singular Value Decomposition)
   - SVD++ 
   - KNN-based factorization

### Evaluation Metrics
- **Ranking Metrics**: NDCG, MAP, Precision@K, Recall@K
- **Rating Prediction**: RMSE, MAE
- **Ranking Loss**: LR, LN-Rank
- **Coverage & Diversity**: Item coverage, similarity diversity

## 📊 Dataset

- **Source**: Amazon Reviews 2023 (Movies & TV category)
- **Initial size**: 500,000 reviews (configurable)
- **After k-core filtering**: Sparse user-item matrix
- **Features**: User ID, Item ID, Rating (1-5), Timestamp, Product Title

### Data Statistics
- Ratings: 1-5 scale
- Sparse matrix representation
- Long-tail distribution in item popularity
- Temporal data available for time-aware recommendations

## 🛠️ Tech Stack

- **Python 3.x**
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Scikit-Learn**: Similarity metrics and utilities
- **Surprise**: Collaborative filtering algorithms
- **Matplotlib**: Visualization
- **JSON/Gzip**: Data format handling

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas matplotlib scikit-surprise
```

### Usage

1. **Load Data**
   - Place `Movies_and_TV.jsonl.gz` in project directory
   - Notebook automatically loads and preprocesses data

2. **Run Analysis**
   - Execute notebook cells sequentially
   - Adjustable parameters: `MAX_LINES`, `SEED`, rating thresholds

3. **Generate Recommendations**
   - Query any user ID for top-K recommendations
   - Supports multiple algorithm selection

## 📈 Key Implementation Details

### Similarity Computation
- Pearson correlation for rating patterns
- Cosine similarity for vector spaces
- Adjustable neighborhood sizes

### Matrix Factorization
- Latent factor learning (SVD, SVD++)
- Configurable regularization
- Convergence optimization

### Evaluation Framework
- Train-test split (80-20 or configurable)
- Cross-validation support
- Ranking & rating evaluation
- Statistical significance testing

## 📊 Results & Analysis

The project includes:
- Comparative performance analysis across algorithms
- Sparsity impact on recommendations
- Popular items distribution visualization
- User-item interaction patterns
- Algorithm efficiency comparison

## 🔍 What's Explored

- ✅ Data quality and preprocessing impact
- ✅ Cold-start problem mitigation strategies
- ✅ Baseline effectiveness as reference point
- ✅ Similarity metric trade-offs
- ✅ Recommendation accuracy vs diversity
- ✅ Scalability considerations
- ✅ Real-world dataset challenges

## 📝 Project Structure

```
├── Recommender_starter latest version).ipynb
├── meta_Movies_and_TV.jsonl          # Metadata file
├── Movies_and_TV.jsonl.gz            # Review data (download)
└── README.md                         # This file
```

## 💡 Future Enhancements

- Content-based features integration
- Context-aware recommendations (time, device)
- Deep learning approaches (neural collaborative filtering)
- Real-time recommendation serving
- Explainability and transparency
- A/B testing framework

## 📄 License

Open source project for educational purposes.
Created as part of advanced data science coursework.
