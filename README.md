<div align="center">

# 🎬 Movie Rating Prediction

### *Predicting IMDb Ratings Using Machine Learning*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-22c55e?style=for-the-badge)

<br/>

> **Can a machine predict how the world will rate a movie before audiences do?**  
> This project explores that question using IMDb's Top 1000 movies dataset,  
> metadata-driven features, and a production-ready sklearn Pipeline.

<br/>

---

</div>

## 📖 Table of Contents

- [About the Project](#-about-the-project)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Exploratory Data Analysis](#-exploratory-data-analysis--key-insights)
- [Model Development](#-model-development)
- [Results](#-results)
- [Live Prediction Test](#-live-prediction-test)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Key Takeaways](#-key-takeaways)
- [Future Scope](#-future-scope)
- [Author](#-author)

---

## 🎯 About the Project

This project builds a **machine learning model** to predict the **IMDb rating** a movie is likely to receive, based purely on its metadata — genre, runtime, release year, critic score, and number of votes.

Rather than collaborative filtering (which requires user-movie interaction matrices), this project takes a **regression-based approach**, making it deployable even for new movies with no user ratings yet.

Two models were built and compared inside a clean **scikit-learn Pipeline**:

| Model | Approach |
|---|---|
| 🔵 Linear Regression | Baseline — fast, interpretable |
| 🌲 Random Forest Regressor | Ensemble — captures non-linear patterns |

---

## 📦 Dataset

**Source:** IMDb Top 1000 Movies (`imdb_top_1000.csv`)

| Property | Value |
|---|---|
| Rows | ~1,000 movies |
| Columns | 16 original → 7 after preprocessing |
| Target | `IMDB_Rating` (continuous, 7.6 – 9.3) |

### Columns Used for Modeling

| Feature | Type | Description |
|---|---|---|
| `Released_Year` | Numeric | Year the movie was released |
| `Runtime` | Numeric | Movie duration in minutes |
| `Genre` | Categorical | Movie genre(s) |
| `Meta_score` | Numeric | Critic score (Metacritic) |
| `No_of_Votes` | Numeric | Total IMDb user votes |
| `IMDB_Rating` | **Target** | User rating on IMDb (1–10) |

### Columns Dropped & Why

| Column | Reason |
|---|---|
| `Poster_Link` | Image URL — not useful for ML |
| `Series_Title` | Identifier only |
| `Certificate` | Low predictive impact |
| `Overview` | Requires NLP — out of scope |
| `Director`, `Star1–4` | Too many unique values (high cardinality) |

---

## 🔄 Project Workflow

```
Raw CSV Data
     │
     ▼
Data Cleaning & Type Fixes
(Runtime → int, Released_Year → Int64, Gross → float)
     │
     ▼
Null Value Handling
(Meta_score filled with column mean)
     │
     ▼
Exploratory Data Analysis
(8 visual insights across ratings, genre, volume, gross)
     │
     ▼
Feature Selection & Train-Test Split (80/20)
     │
     ▼
sklearn Pipeline
├── OneHotEncoder (Genre)
├── StandardScaler
└── Model (Linear Regression / Random Forest)
     │
     ▼
Evaluation (MAE, MSE, R²)
     │
     ▼
Live Prediction on Custom Sample
```

---

## 📊 Exploratory Data Analysis — Key Insights

### 🏆 1. The Golden Era of Cinema
> **1994–1995** produced the highest-rated films on IMDb of all time — *The Shawshank Redemption* (9.3), *Pulp Fiction* (8.9). These films have held their top spots for decades.

### 🎭 2. Drama Rules Everything
> **Drama** dominates the top 10 genres by both highest rating and highest vote count. Pure Action or Comedy genres are almost entirely absent from the top tier — dramatic storytelling drives critical acclaim.

### 🗳️ 3. Most Voted Genre
> Drama attracted over **2.3 million votes** at its peak. Almost every genre in the Top 10 contains "Drama" as a component — audiences engage most deeply with dramatic films.

### 💰 4. Modern Films Earn More, Old Films Rate Better
> **2015** was the highest-grossing year in the dataset. Yet older films (1990s) consistently dominate ratings — a clear inverse relationship between box office revenue and IMDb rating.

### 🔥 5. Critics vs. Audiences — A Weak Agreement
> `Meta_score` and `IMDB_Rating` correlate at only **0.27** — critics and audiences frequently disagree. Emotional experience vs. technical craft are being measured differently.

### 📊 6. Votes Drive Ratings
> `No_of_Votes` has the **strongest correlation (0.55)** with `IMDB_Rating` — more popular movies rate higher, confirming selection bias in voluntary rating systems.

### ⏱️ 7. Longer Films Score Higher
> The top 15 runtimes by IMDb rating are all between **140–202 minutes**. Audiences reward depth and storytelling time — though a tight 96-minute film can still hit 9.0.

### 🌡️ 8. Full Correlation Summary

| Pair | Correlation |
|---|---|
| `No_of_Votes` ↔ `IMDB_Rating` | **+0.55** |
| `No_of_Votes` ↔ `Gross` | **+0.57** |
| `Runtime` ↔ `IMDB_Rating` | **+0.25** |
| `Meta_score` ↔ `IMDB_Rating` | **+0.27** |
| `Released_Year` ↔ `IMDB_Rating` | **−0.18** |

---

## 🤖 Model Development

### Pipeline Architecture

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Preprocessing
trf1 = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Genre'])
], remainder='passthrough')

# Pipeline 1 — Linear Regression
pipe1 = Pipeline([('encode', trf1), ('scale', StandardScaler()), ('model', LinearRegression())])

# Pipeline 2 — Random Forest
pipe2 = Pipeline([('encode', trf1), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
```

---

## 📏 Results

| Metric | Linear Regression | Random Forest | Winner |
|---|---|---|---|
| **MAE** | 0.177 | 0.173 | 🌲 RF |
| **MSE** | 0.050 | 0.048 | 🌲 RF |
| **R² Score** | 0.42 | 0.44 | 🌲 RF |

### What These Numbers Mean

- ✅ **MAE ~0.17** → predictions are off by only **±0.18 stars** on a 10-point scale
- ✅ **Random Forest** edges out Linear Regression across all metrics
- ⚠️ **R² ~0.44** is moderate but expected — there are no individual user preferences in the dataset, and subjective factors like story quality cannot be captured numerically

> 💡 *The bottleneck is the dataset, not the model. Both algorithms perform similarly because the signal ceiling is the same.*

---

## 🧪 Live Prediction Test

```python
import pandas as pd

sample = pd.DataFrame([{
    'Released_Year': 2010,
    'Runtime': 148,
    'Meta_score': 74,
    'No_of_Votes': 2000000,
    'Genre': 'Action'
}])

prediction = pipe.predict(sample)[0]
print(f"Predicted IMDB Rating: {round(prediction, 1)}")
```

```
Predicted IMDB Rating: 8.7
```

> 🎯 This matches the profile of **Inception (2010)** — its actual IMDb rating is **8.8**.  
> The model was off by only **0.1 stars**. 

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Pandas | 2.0 | Data manipulation |
| NumPy | 1.24 | Numerical operations |
| Matplotlib | 3.7 | Visualizations |
| Seaborn | 0.12 | Heatmaps & styled plots |
| Scikit-learn | 1.3 | Pipelines, models, metrics |
| Jupyter Notebook | — | Development environment |

---

## 📁 Project Structure

```
movie-rating-prediction/
│
├── 📓 movie_rating.ipynb       # Main notebook (EDA + modeling)
├── 📄 imdb_top_1000.csv        # Dataset
├── 📋 README.md                # This file
└── 📦 requirements.txt         # Dependencies
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/movie-rating-prediction.git
cd movie-rating-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Notebook
```bash
jupyter notebook movie_rating.ipynb
```

### requirements.txt
```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
jupyter
```

---

## 💡 Key Takeaways

- 🎭 **Drama dominates** IMDb in both ratings and votes — by a wide margin
- 🗳️ **`No_of_Votes`** is the most powerful predictor of a high IMDb rating
- 🎬 **1994–1995** remains the greatest era of cinema by IMDb standards
- 🤝 **Critics and audiences rarely agree** — `Meta_score` is a weak signal
- ⏱️ **Longer films tend to rate higher** — audiences value depth over brevity
- 🌲 **Random Forest** marginally outperforms Linear Regression on this dataset

---

## 🔭 Future Scope

| Enhancement | Description |
|---|---|
| 🔄 Collaborative Filtering | Use MovieLens dataset with user-movie interaction matrix |
| 🧠 NLP on Reviews | Sentiment analysis of `Overview` and user reviews |
| 📈 More Data | Expand beyond Top 1000 to full IMDb catalog |
| 🏷️ Multi-label Genre Encoding | Handle composite genres like "Drama, Crime" properly |
| ⚡ XGBoost / Neural Network | Try gradient boosting or deep learning for better R² |
| 🌐 Web App | Deploy model as a Flask/Streamlit rating predictor |

---

## 👩‍💻 Author

<div align="center">

**Anamta Saleem**  
*Data Science*

</div>

---

<div align="center">

*⭐ If you found this project helpful, consider giving it a star!*


</div>
