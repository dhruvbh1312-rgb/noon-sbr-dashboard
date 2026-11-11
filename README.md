# Noon: Data-Driven Growth Dashboard (SBR project)

A Streamlit dashboard and analysis pipeline for three Noon problems, built on the 9 *Data Driven Growth* blogs.

| Problem | Blogs used | What the dashboard shows |
|---|---|---|
| 1. Cash on delivery (COD) refusals | 1, 2, 4, 8, 9 | Refusal-risk model, risk tiers, savings what-if, prepaid cashback targeting |
| 2. Price-comparing customers switching to rivals | 1, 2, 3, 4, 7, 8, 9 | Churn model, market response uplift, net-value offer targeting what-if |
| 3. Demand spikes in Ramadan and Yellow Friday | 1, 2, 5, 6, 9 | Event-aware forecast vs current plan, stock-out what-if |

**All data is synthetic.** It is generated to behave like a Middle East marketplace (UAE, KSA, Egypt) and is not real Noon data.

Team: Harshita (CEO), Dhruv P Bhansali (Data and Analytics), Abhinav Sharma (Marketing), Shristi Murarka (Customer Experience), Sashidhar (Supply Chain and Operations), Durga Venu (Finance).

## Live operations page

The first page of the dashboard is a **live feed** that updates every 5 seconds in Dubai time:

- **New data every second:** orders, COD deliveries resolving 15 to 60 minutes later, customer reviews, competitor prices and stock levels.
- **Realistic noise:** reviews in English, Arabic and Arabizi with typos, emojis, sarcasm, one-word reviews and fake or spam reviews. Order records arrive with missing cities, negative values and duplicates that a cleaning step fixes.
- **Random real-world events:** flash sales, courier problems in a city and rival price drops. The alert engine is never told about them and has to detect them from the data. You can also trigger one yourself during a presentation.
- **Models running live:** the refusal-risk model scores every new COD order, and a multilingual analyser tags each review's sentiment and topic and flags spam. The page tracks the live ROC-AUC and analyser accuracy.

The stream lives in `src/live_stream.py`. Run `python src/live_stream.py` to watch it in a terminal.

## Run the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Re-run the analysis

```bash
pip install -r requirements-analysis.txt
python src/generate_data.py   # writes data/
python src/analysis.py        # writes results/ and charts/
```

`analysis.py` uses **XGBoost** and a **Keras LSTM** automatically when they are installed. Without them it falls back to scikit-learn HistGradientBoosting and an MLP on the same lag features.

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Go to share.streamlit.io, sign in with GitHub, click **Create app**.
3. Pick the repository, branch `main`, main file `app.py`, then **Deploy**.

## Folder layout

```
app.py                     Streamlit dashboard
src/generate_data.py       synthetic data generator
src/analysis.py            all 9 blog techniques + hyperparameter tuning
src/live_stream.py         real-time event simulator, live models and alert engine
data/                      synthetic input data
results/                   model outputs used by the dashboard
charts/                    static charts used in the report
```
