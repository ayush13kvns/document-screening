import argparse
import json
from pathlib import Path
from pydantic import ValidationError
from .schemas import Document
from .model import Model
from .service import ScreeningService

def run(input_path, output_path, model_path, chunk_size=100, low=0.1, high=0.9, min_coverage=0.6):
    if Path(input_path).resolve()==Path(output_path).resolve():
        raise ValueError("Input and output paths must differ")
    if not 1<=chunk_size<=100: raise ValueError("chunk_size must be 1..100")
    service=ScreeningService(Model(model_path),low,high,min_coverage)
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    # Streaming: every input line receives a result or a redacted validation error.
    with open(input_path,encoding="utf-8-sig") as source, open(output_path,"w",encoding="utf-8") as sink:
        buffer=[]
        def flush():
            if buffer:
                for line,result in zip([v[0] for v in buffer],service.predict([v[1] for v in buffer])):
                    sink.write(json.dumps({"line":line,**result})+"\n")
                buffer.clear()
        line_no=0
        while True:
            line=source.readline(1_000_001)
            if not line:break
            line_no+=1
            if len(line)>1_000_000:
                while not line.endswith("\n"):
                    line=source.readline(1_000_001)
                    if not line:break
                flush(); sink.write(json.dumps({"line":line_no,"error":"INPUT_LINE_TOO_LARGE"})+"\n"); continue
            try:
                doc=Document.model_validate(json.loads(line))
                buffer.append((line_no,doc))
                if len(buffer)>=chunk_size: flush()
            except (ValueError,ValidationError):
                flush()
                sink.write(json.dumps({"line":line_no,"error":"INVALID_DOCUMENT"})+"\n")
        flush()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="data/sample_documents.jsonl")
    p.add_argument("--output",default="outputs/predictions.jsonl")
    p.add_argument("--model",default="artifacts/model.json")
    p.add_argument("--low",type=float,default=0.1)
    p.add_argument("--high",type=float,default=0.9)
    p.add_argument("--min-coverage",type=float,default=0.6)
    a=p.parse_args(); run(a.input,a.output,a.model,low=a.low,high=a.high,min_coverage=a.min_coverage)
    print(f"Results written to {a.output}")

if __name__=="__main__": main()
