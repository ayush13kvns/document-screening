"""Local UTF-8 text, RFC email body, and structured form ingestion."""
import argparse
import hashlib
import json
from email import policy
from email.parser import BytesParser
from pathlib import Path
from .schemas import Document

MAX_BYTES=1_000_000

def extract(path,root):
    if path.is_symlink(): raise ValueError("SYMLINK_NOT_ALLOWED")
    if path.stat().st_size>MAX_BYTES: raise ValueError("FILE_TOO_LARGE")
    raw=path.read_bytes()
    relative=str(path.relative_to(root))
    identifier=hashlib.sha256(relative.encode()).hexdigest()[:24]
    if path.suffix.lower()==".json":
        data=json.loads(raw.decode("utf-8-sig"))
        return Document.model_validate(data)
    if path.suffix.lower()==".eml":
        message=BytesParser(policy=policy.default).parsebytes(raw)
        body=message.get_body(preferencelist=("plain","html"))
        text=str(message.get("Subject",""))+"\n"+(body.get_content() if body else "")
        return Document(id=identifier,text=text,source="email")
    return Document(id=identifier,text=raw.decode("utf-8-sig"),source="ticket")

def ingest(folder,output,errors):
    root=Path(folder).resolve()
    if not root.is_dir(): raise ValueError("Input folder does not exist")
    if Path(output).resolve()==Path(errors).resolve():raise ValueError("Output and error paths must differ")
    for p in (output,errors):
        if Path(p).resolve().is_relative_to(root):
            raise ValueError("Write outputs outside the input folder")
        Path(p).parent.mkdir(parents=True,exist_ok=True)
    count=failed=0
    with open(output,"w",encoding="utf-8") as sink,open(errors,"w",encoding="utf-8") as bad:
        for path in sorted(root.rglob("*")):
            if not path.is_file():continue
            try:
                if path.suffix.lower() not in {".txt",".eml",".json"}:
                    raise ValueError("UNSUPPORTED_FILE_TYPE")
                doc=extract(path,root)
                sink.write(doc.model_dump_json()+"\n");count+=1
            except (ValueError,UnicodeError,LookupError,OSError,TypeError):
                # Record path only; file contents/validation input are never recorded.
                bad.write(json.dumps({"file":str(path.relative_to(root)),"error":"EXTRACTION_FAILED_OR_UNSUPPORTED"})+"\n");failed+=1
    return {"ingested":count,"failed":failed}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--folder",required=True)
    parser.add_argument("--output",default="outputs/ingested.jsonl")
    parser.add_argument("--errors",default="outputs/ingestion_errors.jsonl")
    args=parser.parse_args()
    print(json.dumps(ingest(args.folder,args.output,args.errors)))

if __name__=="__main__":main()
