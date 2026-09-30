import argparse
import csv
import hashlib
import json
import os
import tempfile
import platform
import sklearn
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, confusion_matrix, precision_recall_fscore_support
from scipy.special import expit
from .model import normalize, TOKEN_PATTERN

LABELS = {"legitimate":0, "incorrect":1}

def load_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"id","text","label","group_id","split"} <= set(reader.fieldnames or []):
            raise ValueError("CSV requires id,text,label,group_id,split")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty training data")
    ids, texts, groups = set(),set(),{}
    for row in rows:
        if row["label"] not in LABELS or row["split"] not in {"train","calibration","test"}:
            raise ValueError("Invalid label or split")
        if not row["id"] or row["id"] in ids or not row["group_id"]:
            raise ValueError("IDs must be unique and group IDs nonempty")
        text = normalize(row["text"])
        if len(text.split())<5 or len(row["text"])>30000:
            raise ValueError("Training text must contain >=5 words and <=30000 characters")
        if text in texts:
            raise ValueError("Duplicate normalized text; remove duplicates before assigning splits")
        if row["group_id"] in groups and groups[row["group_id"]] != row["split"]:
            raise ValueError("Group leakage across splits")
        ids.add(row["id"]); texts.add(text); groups[row["group_id"]]=row["split"]
    for split in ("train","calibration","test"):
        counts = Counter(row["label"] for row in rows if row["split"]==split)
        if any(counts[label]<5 for label in LABELS):
            raise ValueError(f"{split} requires >=5 examples per label; real deployment needs far more")
    return rows

def atomic_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp = tempfile.mkstemp(dir=path.parent,suffix=".tmp")
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def train(data_path, model_path, report_path, demo=False, low=0.1, high=0.9):
    if not 0<=low<high<=1: raise ValueError("Invalid thresholds")
    rows=load_rows(data_path)
    sets={s:[r for r in rows if r["split"]==s] for s in ("train","calibration","test")}
    texts=lambda s:[normalize(r["text"]) for r in sets[s]]
    labels=lambda s:np.array([LABELS[r["label"]] for r in sets[s]])
    vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=50000,token_pattern=TOKEN_PATTERN)
    x=vectorizer.fit_transform(texts("train"))
    clf=LogisticRegression(C=2,max_iter=1000,random_state=42)
    clf.fit(x,labels("train"))
    # Sigmoid calibration on a disjoint split; test labels never tune this mapping.
    scores=clf.decision_function(vectorizer.transform(texts("calibration")))
    calibration=LogisticRegression(C=1,max_iter=1000,random_state=42)
    calibration.fit(scores.reshape(-1,1),labels("calibration"))
    test_scores=clf.decision_function(vectorizer.transform(texts("test")))
    p=expit(test_scores*calibration.coef_[0,0]+calibration.intercept_[0])
    y=labels("test"); pred=(p>=0.5).astype(int)
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
    selected=(p<=low)|(p>=high)
    report={"demo":demo,"warning":"Synthetic results do not establish real-world accuracy" if demo else "Independent domain validation required",
            "counts":{s:dict(Counter(r["label"] for r in sets[s])) for s in sets},
            "test_metrics":{"pr_auc_average_precision":float(average_precision_score(y,p)),"roc_auc":float(roc_auc_score(y,p)),
            "precision_at_0_5":float(precision),"recall_at_0_5":float(recall),"f1_at_0_5":float(f1),
            "brier_score":float(brier_score_loss(y,p)),"confusion_matrix_true_rows_pred_columns":confusion_matrix(y,pred,labels=[0,1]).tolist(),
            "score_only_coverage":float(selected.mean()),"score_only_review_rate":float(1-selected.mean()),
            "selected_accuracy":float((pred[selected]==y[selected]).mean()) if selected.any() else None},
            "thresholds":{"low":low,"high":high},
            "data_sha256":hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
            "notes":"Selective metrics use score thresholds only; operational decisions also apply text quality, demo and evidence gates."}
    artifact={"schema_version":1,"demo":demo,"vocabulary":{k:int(v) for k,v in vectorizer.vocabulary_.items()},
              "idf":vectorizer.idf_.tolist(),"coef":clf.coef_[0].tolist(),"intercept":float(clf.intercept_[0]),
              "calibration_coef":float(calibration.coef_[0,0]),"calibration_intercept":float(calibration.intercept_[0]),
              "data_sha256":report["data_sha256"],"training_python":platform.python_version(),"training_sklearn":sklearn.__version__}
    atomic_json(model_path,artifact); atomic_json(report_path,report)
    return report

def main():
    parser=argparse.ArgumentParser(description="Train document screening model")
    parser.add_argument("--data",default="data/demo_training.csv")
    parser.add_argument("--model",default="artifacts/model.json")
    parser.add_argument("--report",default="artifacts/evaluation.json")
    parser.add_argument("--demo",action="store_true",help="Mark synthetic model; disables automatic text decisions")
    parser.add_argument("--low",type=float,default=0.1)
    parser.add_argument("--high",type=float,default=0.9)
    args=parser.parse_args()
    # Bundled data can never silently be treated as a production model.
    bundled=Path(__file__).resolve().parents[1]/"data/demo_training.csv"
    demo=args.demo or (bundled.exists() and hashlib.sha256(Path(args.data).read_bytes()).digest()==hashlib.sha256(bundled.read_bytes()).digest())
    report=train(args.data,args.model,args.report,demo=demo,low=args.low,high=args.high)
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
