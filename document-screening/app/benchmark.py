"""Local throughput measurement; no network/GPU and no invented capacity claim."""
import argparse
import json
import time
from .model import Model
from .service import ScreeningService
from .schemas import Document

def main():
    p=argparse.ArgumentParser();p.add_argument("--model",default="artifacts/model.json")
    p.add_argument("--documents",type=int,default=10000)
    a=p.parse_args()
    if not 1<=a.documents<=1000000:p.error("documents must be 1..1000000")
    service=ScreeningService(Model(a.model))
    docs=[Document(id=str(i),text="The invoice details match the approved customer registration and verified official billing record") for i in range(100)]
    service.predict(docs)  # Warm-up excluded.
    start=time.perf_counter()
    for offset in range(0,a.documents,100):service.predict(docs[:min(100,a.documents-offset)])
    elapsed=time.perf_counter()-start
    print(json.dumps({"documents":a.documents,"seconds":round(elapsed,4),"documents_per_second":round(a.documents/elapsed,2),"note":"In-process synthetic short-text inference only; excludes extraction, API, storage and real-document variability"},indent=2))

if __name__=="__main__":main()
