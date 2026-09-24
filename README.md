# Selangor Flood Prediction from Satellite Rainfall

A machine-learning system that predicts whether a flood will be recorded in Selangor, Malaysia on a given day, using NASA satellite rainfall over the state and ten years of official flood records. It was built as the final-year project of the Diploma in Computer Science at Sunway College (Dec 2022 to Apr 2023) and restored to GitHub in 2026, with the data pipeline migrated to the current NASA product and a corrected evaluation added.

| | |
|---|---|
| **Team** | Ian Leong Zheng Yan and Hunts Wong Han Xian |
| **Supervisor** | Mr Hasbullah Osman, Sunway Diploma Studies |
| **Unit** | DIT2166 IT Project, Diploma in Computer Science, Sunway College |
| **Original period** | December 2022 to April 2023 |
| **Restored** | September 2026 |

## What it does

Every year Selangor, the most industrialised state in Malaysia, suffers monsoon and flash floods. Flooding in 2021 alone caused about RM 3.1 billion in damage. The project asks a simple question: can satellite rainfall over the state predict whether a flood will be recorded that day?

1. **Rainfall.** NASA's GPM IMERG product gives daily rainfall on a 0.1 degree grid. A 15 x 15 window over Selangor (225 cells) is pulled for every day from 2001 to 2010.
2. **Labels.** The Department of Irrigation and Drainage (JPS) publishes flood records by river basin. These were cleaned into a daily flood or no-flood flag for seven Selangor rivers (Bernam, Tengi, Selangor, Buloh, Klang, Langat, Sepang) and collapsed into one label per day.
3. **Models.** Four classifiers are trained on the 225 rainfall cells: Decision Tree, Support Vector Machine, Logistic Regression and Gaussian Naive Bayes, with SMOTE to handle the class imbalance. This is done twice, once for each IMERG rainfall estimate.
4. **Web app.** A Flask app fetches the latest day of rainfall from NASA on startup, runs all eight trained models and draws the rainfall grid as a heatmap over a map of Selangor with each model's prediction.

## Results (2023, as submitted)

Test set of 2,006 days after SMOTE, 70/30 random split. Two rainfall products were compared: the microwave-only estimate (called HQprecipitation in 2023, MWprecipitation in V07) and the calibrated multi-satellite estimate (precipitationCal, now precipitation).

| Model | HQ accuracy | HQ flood F1 | Calibrated accuracy | Calibrated flood F1 |
|---|---|---|---|---|
| Decision Tree | 84.8% | 0.85 | 84.7% | 0.85 |
| Support Vector Machine | 83.0% | 0.82 | 83.1% | 0.82 |
| Logistic Regression | 71.2% | 0.69 | 73.4% | 0.72 |
| Naive Bayes | 60.5% | 0.48 | 62.5% | 0.53 |

The full confusion matrices are saved in the notebooks, which keep their 2023 outputs. Both notebooks also run the trained models on December 2021 and June 2021 as informal hold-out months.

**These numbers are inflated.** See [Known limitations](#known-limitations-and-the-2026-fix). They are kept here because they are what was submitted and marked.

## Repository layout

```
selangor-flood-prediction/
├── README.md
├── requirements.txt
├── data/
│   ├── Flood_Events.xlsx                  daily flood flags for 7 rivers, 2001-2010 (cleaned)
│   ├── JPS_flood_records_2000-2010.xls    the official source records
│   └── samples/                           one day of rainfall in each format, for testing
├── notebooks/
│   ├── Flood_Model_Comparison_HQprecip.ipynb     2023 training runs, outputs included
│   └── Flood_Model_Comparison_PrecipCal.ipynb
├── scripts/
│   ├── imerg_client.py        V07 client: URL builder, Earthdata session, NetCDF extraction
│   ├── imerg_download.py      resumable bulk download for any date range
│   └── train_v2.py            corrected evaluation (2026), see below
├── webapp/
│   ├── app.py                 Flask app (2023 code, paths updated)
│   ├── imerg_scraper.py       fetches the latest day (V07)
│   ├── models/                the eight 2023 trained models (.pkl)
│   ├── templates/index.html
│   └── SelangorMap.png
├── legacy_2023/               the original scraper, loader and app, untouched
└── docs/Project_Proposal_Dec2022.pdf
```

## Setup

```bash
git clone https://github.com/imosleo/selangor-flood-prediction.git
cd selangor-flood-prediction
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

NASA GES DISC requires a free [Earthdata Login](https://urs.earthdata.nasa.gov/). Create the account, approve the application **NASA GESDISC DATA ARCHIVE** under Applications > Authorized Apps, and save your credentials in `~/.netrc` (Windows: `%USERPROFILE%\_netrc`):

```
machine urs.earthdata.nasa.gov
    login    YOUR_USERNAME
    password YOUR_PASSWORD
```

### Run the web app

```bash
cd webapp
python app.py
```

Open http://127.0.0.1:5000. On startup the app checks whether its cached rainfall is from the latest available day and, if not, downloads it from NASA. Choose a rainfall product to see the heatmap and a model to see its prediction.

Six of the eight 2023 models load under current scikit-learn. The two Decision Trees were pickled with scikit-learn 1.2 and need an environment pinned to that version, or a retrain with `train_v2.py`.

### Rebuild the training data

The 2001 to 2010 rainfall CSVs used in 2023 were not kept, and the V06 files they came from have been retired by NASA. Re-download them under V07:

```bash
python scripts/imerg_download.py                 # 3,652 days, about 1 to 3 hours
```

The script is resumable and lists any failed days for a second pass. It writes `data/HQprecipitation Data.csv` and `data/precipitationCal Data.csv` in the exact layout the notebooks read.

## Known limitations and the 2026 fix

The 2023 evaluation has two flaws that inflate the results above.

- **SMOTE before the split.** Synthetic flood days generated from the training data were also placed in the test set, so the models were partly tested on copies of what they trained on.
- **Random split of daily weather.** Consecutive days share rainfall, so a random split lets the model see near-duplicates of test days during training. Time-series data needs a chronological split.

There is also a design limit: the models predict a flood from the same day's rainfall, which is closer to nowcasting than forecasting.

`scripts/train_v2.py` keeps the same four models and data and corrects all three points:

| | 2023 notebooks | train_v2.py |
|---|---|---|
| Split | random 70/30 | chronological, train 2001-2008, test 2009-2010 |
| SMOTE | whole dataset | training fold only |
| Features | today's 225 cells | today plus the previous 3 days (900) |
| Scaling | none | StandardScaler for Logistic Regression and SVM |
| Metrics | accuracy | precision, recall and F1 for the flood class |

```bash
python scripts/train_v2.py --synthetic                                   # pipeline smoke test, no data needed
python scripts/train_v2.py --rain "data/HQprecipitation Data.csv" --col HqPrecips
python scripts/train_v2.py --rain "data/precipitationCal Data.csv" --col PrecipCals
```

The corrected numbers will be lower than the 2023 table, and that is the point. The results table will be updated once the V07 re-download has finished.

## What changed from the 2023 submission

- **NASA IMERG V06 was retired** and replaced by V07, which renamed the two rainfall variables. `scripts/imerg_client.py` and `webapp/imerg_scraper.py` are the V07 versions of the original scraper and loader; the originals are kept in `legacy_2023/`. The grid, output format and models are unchanged.
- `webapp/app.py` now loads models from `models/` and resolves paths relative to itself. The logic is the 2023 code.
- `scripts/imerg_download.py` replaces the manual GES DISC subset list the loader used, and `scripts/train_v2.py` is new.
- The notebooks are the 2023 files with their saved outputs, renamed only.

## Data sources

- NASA GPM IMERG Late Precipitation L3 1 day 0.1 degree V07 (GPM_3IMERGDL), GES DISC. https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGDL_07/summary
- Jabatan Pengairan dan Saliran Malaysia (JPS), *Rekod Banjir 2000 hingga 2010*, flood records by river basin.
