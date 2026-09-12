# TruLens — Fake News Detection (Demo)

TruLens is a lightweight demo application for detecting likely fake news in short text. It's built with Python and Flask and combines simple text features (TF-IDF, word counts) with sentiment metrics and a Logistic Regression classifier. The app provides a small web UI to paste or type an article and get a prediction, confidence score, and diagnostic features.

This repository is intended as an educational/demo project — it shows how to combine basic NLP preprocessing, feature engineering, and a simple ML model in a web app. It is not production-ready and should not be used as-is for high-stakes decision making.

## Table of contents

- Features
- Quickstart (Windows cmd examples)
- Installation
- Usage (web and CLI)
- Project layout
- Development notes
- Testing and verification
- Troubleshooting
- Contributing
- License

## Features

- Detects likely fake vs real news using a pretrained Logistic Regression model
- Displays prediction probability/confidence
- Shows simple diagnostic features: sentiment (polarity & subjectivity), word counts, average word length
- Minimal web interface served with Flask (see `templates/index.html`)
- Includes a small scraper and sample JSON dataset (`news_data.json`) for demo purposes

## Quickstart (Windows cmd)

1. Create and activate a virtual environment (recommended):

```cmd
python -m venv .venv
.venv\Scripts\activate
```

1. Install dependencies:

```cmd
pip install -r requirements.txt
```

1. Run the web app:

```cmd
python app.py
```

1. Open your browser at: `http://127.0.0.1:5000/`

Notes: If you prefer the Streamlit version, run `streamlit run streamlit_app.py` after installing `streamlit`.

## Installation

1. Clone or download this repository.
2. (Optional) Use a virtual environment as shown in Quickstart.
3. Install required packages:

```cmd
pip install -r requirements.txt
```

If any package installation fails on Windows, ensure you have a compatible Python version (3.8–3.11 recommended) and build tools installed for packages with native extensions.

## Usage

Web UI (Flask):

- Start the app with `python app.py`.
- Open the browser and paste a news article into the text area.
- Click "Detect Fake News" and review the prediction, probability, and diagnostics.

Streamlit UI (optional):

- Install Streamlit (`pip install streamlit`) or ensure it's listed in `requirements.txt`.
- Run `streamlit run streamlit_app.py`.

Command-line / Developer:

- `scraper.py` is a small script used to fetch or assemble demo articles into `news_data.json` (see docstring in the file).
- The model and vectorizer artifacts may be created on first run and saved as `fake_news_model.pkl` and `tfidf_vectorizer.pkl` in the project root.

## Project layout

- `app.py` — Flask app and endpoints for the web UI.
- `streamlit_app.py` — Alternative Streamlit frontend (single-file app).
- `scraper.py` — Simple scraping / data preparation utilities used to produce `news_data.json`.
- `news_data.json` — Sample dataset / demo articles included for quick testing.
- `templates/index.html` — Minimal HTML template used by the Flask app.
- `requirements.txt` — Python dependencies for the project.
- `README.md` — This file.
- `LICENSE` — Project license.

If present after a run:

- `fake_news_model.pkl` — Pickled trained classifier.
- `tfidf_vectorizer.pkl` — Pickled TF-IDF vectorizer used during training and inference.

## Development notes

- Model: The demo uses TF-IDF features plus simple numeric features (word counts, avg word length) and sentiment (polarity & subjectivity). A Logistic Regression classifier is trained on a small dataset for illustration.
- Retraining: To retrain, you'll need a labeled dataset. Add a training script that recreates the vectorizer and model and then pickle them to the project root using the same filenames referenced by `app.py`.
- Data handling: `news_data.json` contains a few example articles — expand or replace with a larger, labeled dataset for better results.

### Edge cases & limitations

- Short texts may produce unreliable predictions.
- Model is only as good as the training data; this demo uses a small sample dataset.
- No authentication, rate limiting, or hardening is implemented.

## Testing and verification

Quick checks:

1. Verify Python can import the required packages:

```cmd
python -c "import flask, joblib, sklearn, textblob; print('imports ok')"
```

1. Start the app and submit a few example articles from `news_data.json`.

If you add tests, put them under a `tests/` directory and run them with pytest.

## Troubleshooting

- Missing model/vectorizer files: Run the training routine (if available) or trigger the app's first-run behavior which may create those files. Check `app.py` for how it loads or builds artifacts.
- Port already in use: The Flask app uses the default port 5000. Stop other services or set the `PORT` environment variable before starting.
- Errors installing dependencies: Ensure pip is up-to-date (`python -m pip install --upgrade pip`) and that you have a supported Python version.

### Useful commands (Windows cmd)

```cmd
rem upgrade pip
python -m pip install --upgrade pip

rem activate venv
.venv\Scripts\activate

rem run app
python app.py

rem run streamlit ui
streamlit run streamlit_app.py
```

## Contributing

Contributions are welcome. For small fixes, open a PR with a clear description. For larger changes (model improvements, tests, CI), open an issue first so we can discuss scope and compatibility.

## License

This project is distributed under the terms in the `LICENSE` file in the repository root.

## Acknowledgements

This demo was created for educational purposes to show a basic pipeline combining NLP preprocessing, feature engineering, and a light-weight classifier served via Flask/Streamlit.
