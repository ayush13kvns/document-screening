import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from .model import Model
from .schemas import Batch
from .service import ScreeningService

log=logging.getLogger("screening")
logging.basicConfig(level=logging.INFO)

def create_app():
    @asynccontextmanager
    async def lifespan(app):
        key=os.getenv("API_KEY","")
        if len(key)<24: raise RuntimeError("Set API_KEY to a random secret of at least 24 characters")
        model=Model(os.getenv("MODEL_PATH","artifacts/model.json"))
        if model.demo and os.getenv("ALLOW_DEMO_MODEL","false").lower()!="true":
            raise RuntimeError("Demo model blocked: for local demo only set ALLOW_DEMO_MODEL=true")
        app.state.api_key=key
        app.state.service=ScreeningService(model,float(os.getenv("LOW_THRESHOLD","0.10")),float(os.getenv("HIGH_THRESHOLD","0.90")),float(os.getenv("MIN_COVERAGE","0.60")))
        yield
    app=FastAPI(title="Document Screening API",version="1.0.0",lifespan=lifespan)

    @app.middleware("http")
    async def limits_and_logs(request:Request, call_next):
        request_id=uuid.uuid4().hex; started=time.perf_counter()
        # Check actual body bytes, including chunked input, before Pydantic parsing.
        if request.method in {"POST","PUT","PATCH"}:
            total=0; parts=[]
            async for chunk in request.stream():
                total+=len(chunk)
                if total>1_000_000:
                    return JSONResponse({"detail":"Request body exceeds 1 MB","request_id":request_id},status_code=413)
                parts.append(chunk)
            request._body=b"".join(parts)
        response=await call_next(request)
        response.headers["X-Request-ID"]=request_id
        log.info(json.dumps({"request_id":request_id,"method":request.method,"status":response.status_code,"duration_ms":round((time.perf_counter()-started)*1000,2)}))
        return response

    # Pydantic errors otherwise echo sensitive input in the response.
    from fastapi.exceptions import RequestValidationError
    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse({"detail":[{"location":list(e["loc"]),"type":e["type"]} for e in exc.errors()]},status_code=422)

    def authenticate(request:Request,x_api_key:str=Header(default="")):
        if not hmac.compare_digest(x_api_key,request.app.state.api_key):
            raise HTTPException(status_code=401,detail="Invalid API key")

    @app.get("/",include_in_schema=False)
    def root(): return {"service":"document-screening","docs":"/docs","health":"/health/live"}

    @app.get("/health/live")
    def live(): return {"status":"ok"}

    @app.get("/health/ready")
    def ready(request:Request):
        return {"status":"ready","model_version":request.app.state.service.model.version}

    @app.post("/v1/classify",dependencies=[Depends(authenticate)])
    def classify(batch:Batch,request:Request):
        return {"results":request.app.state.service.predict(batch.documents)}
    return app

app=create_app()
