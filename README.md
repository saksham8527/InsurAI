# Insurance Cost Predictor

A full-stack insurance-charge estimator with a FastAPI API and a React/Vite
dashboard. It uses the supplied insurance dataset and a joblib-serialized
scikit-learn pipeline.

## Project layout

```text
Insurance/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   ├── api.js
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── insurance.csv
├── insurance_model_compat.pkl
├── migrate_model.py
└── README.md
```

The backend loads `insurance.csv` and `insurance_model_compat.pkl` from the
project root. The original model file is also included as `insurance_model .pkl`
(with a space before `.pkl`); `migrate_model.py` can regenerate the compatible
artifact if needed.

## Requirements

- Python 3.10–3.12
- Node.js 20.19+ or 22.12+

## Run the backend

From the project root, create and activate a virtual environment, then install
the backend dependencies:

```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API runs at <http://localhost:8000>. Interactive API documentation is at
<http://localhost:8000/docs>.

## Run the frontend

In a second terminal, from the project root:

```powershell
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The frontend calls the backend at
`http://localhost:8000`; CORS is enabled for the Vite development origin.

## API

- `POST /api/predict` — validates applicant details and returns a charge
  estimate, a ±$2,500 range, risk level, and comparison with the dataset average.
- `GET /api/model/metrics` — held-out R², MAE, RMSE, and dataset summary metrics.
- `GET /api/health` — backend/model availability and uptime.
- `GET /api/charts/charges-by-smoker`
- `GET /api/charts/charges-by-region`
- `GET /api/charts/age-vs-charges`
- `GET /api/charts/bmi-vs-charges`
- `GET /api/charts/charges-distribution`
- `GET /api/charts/feature-importance`
- `GET /api/charts/actual-vs-predicted`

Predictions are for educational purposes and are not an insurance quote.
