# CrediSense — Backend scaffold

This repository contains a scaffolded backend for the CrediSense loan eligibility system.

Quick start (development):

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the FastAPI app with uvicorn:

```bash
uvicorn src.credisense.app:app --reload
```

3. Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d \
'{"income":50000,"loan_amount":200000,"cibil_score":680}'
```

Testing:

```bash
pytest -q
```
