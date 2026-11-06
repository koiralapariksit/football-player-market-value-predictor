# ⚽ Football / Soccer Player Market Value Predictor

A high-performance machine learning system and interactive web application that predicts professional football/soccer player transfer market values in Euros (€) based on real-world player statistics, match performance, and institutional club prestige.

Trained on over **26,182 active professional players** and **1.89+ million match appearances** from the Transfermarkt relational database across Europe and global competitions.

---

## 🌟 Key Features

- **⚡ Fast Name Autocomplete & Search**: Instant accent-insensitive search (<15ms) across 26,000+ players (e.g. typing `lio` immediately surfaces `Lionel Messi` with club, position, and face thumbnail).
- **🤖 High-Accuracy Super-Ensemble Model**: Blends **LightGBM (40%)**, **XGBoost (35%)**, and **Histogram Gradient Boosting (25%)** with 62 engineered domain features.
- **🏛️ Bayesian Club Prestige Indexing**: Quantifies club financial stature across 2,206 clubs, capturing institutional valuation premiums.
- **🌱 Wonderkid Breakout Dynamics**: Speculative valuation curve modeling for emerging talents ($\le 21$ years old) starting in top leagues.
- **🎨 Modern Dark Stadium UI**: Glassmorphic web dashboard with official player portraits, 6-axis Chart.js radar charts, market sentiment badges, and verified stats.
- **🔬 Interactive What-If Scenario Sandbox**: Move live sliders to simulate future seasons (e.g. *What if he scores 15 more goals?* or *What if he transfers to the Premier League?*).

---

## 🚀 Benchmark Performance (Held-Out Test Set: 5,237 Players)

The model targets $\ln(1 + \text{market\_value\_in\_eur})$ to stabilize the extreme heavy-tailed financial distribution of football transfers (€10K to €200M+) and maps outputs back to real Euro values.

| Model Architecture | Test $\text{Log } R^2$ | Raw Euro $R^2$ | Test MAE (€) | Test MedAE (€) | Test MdAPE | Within $\pm 20\%$ | Within $\pm 50\%$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Super Ensemble (LightGBM + XGBoost + HistGBM)** | **0.7684** | **0.7919** | **€1,015,504** | **€178,813** | **41.2%** | **24.6%** | **59.1%** |
| **HistGradientBoosting Regressor** | 0.7276 | 0.8004 | €1,056,359 | €192,219 | 45.3% | 23.8% | 54.2% |
| **Random Forest Regressor** | 0.6929 | 0.6943 | €1,186,983 | €206,897 | 49.2% | 21.2% | 50.7% |
| **Ridge Regression Baseline** | 0.5937 | 0.5967 | €1,395,181 | €249,450 | 57.3% | 17.1% | 44.5% |

- **Median Absolute Error = €178,813**: Half of all test predictions are within €178K of true market values across the global dataset.
- **Within $\pm 50\%$**: **59.1%** of all players are evaluated within half their listed market value.

---

## 🌟 Real Star Player Validation

| Player | Position & Club | Age | Actual Listed Value | Model Predicted Value | 80% Confidence Interval | Alignment |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Cole Palmer** | AM, Chelsea FC | 23 | **€100.0M** | **€98.1M** | €34.8M – €276.7M | **Highly Aligned (-1.9%)** |
| **Bukayo Saka** | RW, Arsenal FC | 24 | **€110.0M** | **€97.6M** | €34.6M – €275.2M | **Highly Aligned (-11.3%)** |
| **Viktor Gyökeres** | CF, Arsenal / Sporting | 27 | **€65.0M** | **€89.6M** | €31.8M – €252.6M | Model Evaluates Higher (+37.8%) |
| **Erling Haaland** | CF, Manchester City | 25 | **€200.0M** | **€157.2M** | €55.8M – €443.4M | Model Evaluates Lower (-21.4%) |
| **Lamine Yamal** | RW, FC Barcelona | 18 | **€200.0M** | **€116.8M** | €41.4M – €329.4M | Model Evaluates Lower (-41.6%) |
| **Lionel Messi** | RW, Inter Miami CF | 38 | **€15.0M** | **€6.1M** | €2.2M – €17.3M | Veteran Age Discount (-59.1%) |

---

## 📂 Repository Structure

```
football-player-market-value-predictor/
├── app/
│   ├── templates/
│   │   └── index.html               # Stadium glassmorphic web dashboard
│   ├── search_index.py              # In-memory accent-insensitive autocomplete engine (<15ms)
│   ├── server.py                    # Flask REST API & Web Server
│   └── web_scraper.py               # Live Transfermarkt profile scraper
├── data/
│   └── processed/
│       └── football_players_market_value.csv.gz # Curated 26k+ player dataset
├── models/
│   ├── best_market_value_model.joblib # Trained Super Ensemble pipeline
│   └── club_prestige_map.json       # Bayesian club prestige indices (2,206 clubs)
├── src/
│   ├── data_loader.py               # ETL pipeline (downloads and joins relational data)
│   ├── ensemble_pipeline.py         # Super-Ensemble estimator class
│   ├── feature_engineering.py       # 62 domain & institutional features
│   ├── train_super_model.py         # LightGBM + XGBoost + HistGBM training pipeline
│   ├── evaluate.py                  # Error diagnostics by position, age, and league
│   └── predict.py                   # High-level Python inference API
├── tests/
│   └── test_pipeline.py             # Automated unit and integration test suite
├── requirements.txt                 # Dependencies
└── README.md
```

---

## 🛠️ Quick Start

### 1. Installation

```bash
git clone https://github.com/koiralapariksit/football-player-market-value-predictor.git
cd football-player-market-value-predictor
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Application

```bash
python app/server.py
```
Open your browser and navigate to **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.
- Type any player name in the search bar (e.g. `messi`, `haaland`, `palmer`, `saka`, `yamal`).
- Click any suggestion to load their portrait, radar spider chart, verified stats, and fair market valuation!

### 3. Command Line Interface (CLI)

```bash
python src/cli.py --player "Cole Palmer"
python src/cli.py --player "Erling Haaland"
python src/cli.py --player "Lionel Messi"
```

Or launch interactive prompt:
```bash
python src/cli.py
```

### 4. Run Automated Test Suite

```bash
python -m unittest tests/test_pipeline.py
```
