from fastapi import FastAPI

app = FastAPI(title="motor-fgi-baseline")

@app.get("/")
def root():
    return {"status": "ok", "service": "motor-fgi-baseline"}

@app.get("/health")
def health():
    return {"status": "ok"}
