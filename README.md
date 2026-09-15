# ThreatLens AI
Evidence-Aware Autonomous Network Security Analyst

## What is here

Member 1's ML pipeline remains in `src/` and `models/`. The unified entry point is
`src.predictor.predict_threat(flow)`, which performs preprocessing, suspicious
traffic classification, anomaly detection, attack-family classification,
deterministic risk scoring, and feature explanation.

Member 2's MVP integration is organized as:

```text
frontend/ React dashboard
		-> backend/ FastAPI
		-> src.predictor SecurityEvent
		-> backend investigation summary and recommendation
		-> alert stream rendered in the dashboard
```

Detected existing ML files include `src/predictor.py`, `src/risk_engine.py`,
`src/explainability.py`, the preprocessing and training modules, serialized
artifacts in `models/`, and evaluation/handoff reports in `outputs/`. There was
no existing backend, frontend, API, or database before this integration.

## Run locally

From the repository root, install Python dependencies and start FastAPI:

```powershell
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173/`. The dashboard can load a demo event or accept a
raw UNSW-NB15-compatible flow as JSON.

## API

- `GET /api/health` confirms the service is running.
- `POST /api/predict` accepts raw feature fields or `{ "flow": { ... } }` and
	returns a typed standardized Security Event from the real ML pipeline.
- `POST /api/investigate` accepts the same input and returns the event plus
	evidence and a recommendation. It also accepts `{ "event_id": "..." }`
	for investigating an event already returned by `/api/predict`.
- `GET /api/alerts` returns recent suspicious Security Events.
- `GET /api/alerts/{event_id}` returns one suspicious event by ID.
- `GET /api/dashboard/stats` returns traffic totals, severity totals, attack
	distribution, and recent alerts.
- `GET /api/sample` runs the documented demo flow through the real ML pipeline.

Risk is never calculated by the investigation layer. It is preserved from the
deterministic `src.risk_engine` output. The alert feed is intentionally
in-memory for this hackathon MVP, so it resets when FastAPI restarts.

The serialized models currently emit scikit-learn compatibility warnings when
loaded under the local environment's older scikit-learn version. Inference was
verified successfully; retraining or changing artifacts is outside this
integration task.

## Investigation agent

The lightweight agent is implemented in `backend/investigation_agent.py`. Its
tools retrieve only registered application data:

- `get_alert(event_id)` retrieves the standardized ML event.
- `get_traffic_details(event_id)` retrieves the original submitted flow.
- `get_ip_history(source_ip)` searches previously registered events.
- `get_attack_evidence(event_id)` formats classifier and anomaly outputs.
- `get_shap_explanation(event_id)` returns predictor-provided explanation features.
- `calculate_risk(event_id)` returns the risk and severity already produced by
	`src.risk_engine` through the predictor.

The agent does not classify traffic, invent history, generate model scores, or
recalculate risk. Missing evidence is returned as explicitly unavailable.
