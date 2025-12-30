from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="motor-fgi baseline", version="0.1.0")

@app.get("/")
def root():
    return {"status": "ok", "service": "motor-fgi"}

@app.get("/health")
def health():
    return JSONResponse(
        status_code=200,
        content={
            "healthy": True,
            "message": "Render is running"
        }
    )
