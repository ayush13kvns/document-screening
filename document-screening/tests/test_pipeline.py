import csv
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.train import train, load_rows
from app.model import Model, normalize
from app.schemas import Document
from app.service import ScreeningService, equal_value
from app.batch import run
from app.api import create_app

DATA=Path(__file__).resolve().parents[1]/"data/demo_training.csv"

@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    root=tmp_path_factory.mktemp("model")
    model=root/"model.json"
    train(DATA,model,root/"report.json",demo=True)
    return model

def doc(**values):
    return Document(id="test",text="The invoice amount matches the verified official billing record",**values)

def test_demo_and_evidence(artifact):
    s=ScreeningService(Model(artifact))
    assert s.predict([doc()])[0]["decision"]=="review"
    result=s.predict([doc(claims={"invoice_total":"100"},reference={"invoice_total":"101"})])[0]
    assert result["decision"]=="likely_incorrect"
    assert result["basis"]=="reference_mismatch"
    assert "101" not in json.dumps(result)

@pytest.mark.parametrize("a,b,expected",[("100.00","100",True),("1,000","1000",None),("NaN","0",None),("-1","1",None),("10","20",False)])
def test_decimal(a,b,expected): assert equal_value("invoice_total",a,b) is expected

def test_short_unknown_and_missing(artifact):
    s=ScreeningService(Model(artifact))
    d=Document(id="short",text="")
    assert s.predict([d])[0]["reasons"]==["SHORT_TEXT"]
    d=doc(claims={"unknown":"x","currency":"INR"})
    statuses={x["status"] for x in s.predict([d])[0]["checks"]}
    assert statuses=={"unsupported","missing_reference"}

def test_normalization():
    assert normalize("<p>HELLO</p> alice@example.com https://example.com")=="hello emailtoken urltoken"

def test_json_inference_matches_training(tmp_path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    from scipy.special import expit
    rows=load_rows(DATA)
    subsets={s:[r for r in rows if r["split"]==s] for s in ("train","calibration","test")}
    v=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=50000)
    clf=LogisticRegression(C=2,max_iter=1000,random_state=42).fit(v.fit_transform([normalize(r["text"]) for r in subsets["train"]]),[r["label"]=="incorrect" for r in subsets["train"]])
    scores=clf.decision_function(v.transform([normalize(r["text"]) for r in subsets["calibration"]]))
    cal=LogisticRegression(C=1,max_iter=1000,random_state=42).fit(scores.reshape(-1,1),[r["label"]=="incorrect" for r in subsets["calibration"]])
    model=tmp_path/"model.json";train(DATA,model,tmp_path/"report.json",True)
    texts=[r["text"] for r in subsets["test"]]
    expected=expit(clf.decision_function(v.transform([normalize(t) for t in texts]))*cal.coef_[0,0]+cal.intercept_[0])
    np.testing.assert_allclose(Model(model).score(texts)[0],expected,rtol=1e-12)

def test_leakage_rejected(tmp_path):
    rows=load_rows(DATA); rows[-1]["group_id"]=rows[0]["group_id"]
    path=tmp_path/"bad.csv"
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match="Group leakage"): load_rows(path)

def test_duplicate_rejected(tmp_path):
    rows=load_rows(DATA);rows[-1]["text"]=rows[0]["text"]
    path=tmp_path/"bad.csv"
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match="Duplicate"): load_rows(path)

def test_streaming_batch_errors(artifact,tmp_path):
    source=tmp_path/"input.jsonl";target=tmp_path/"output.jsonl"
    source.write_text(json.dumps(doc().model_dump())+"\nnot json\n"+json.dumps(doc().model_dump())+"\n")
    run(source,target,artifact,chunk_size=2)
    rows=[json.loads(x) for x in target.read_text().splitlines()]
    assert [x["line"] for x in rows]==[1,2,3]
    assert rows[1]["error"]=="INVALID_DOCUMENT"
    with pytest.raises(ValueError):run(source,source,artifact)

def test_api(artifact,monkeypatch):
    monkeypatch.setenv("MODEL_PATH",str(artifact));monkeypatch.setenv("API_KEY","test-secret-at-least-24-characters")
    monkeypatch.setenv("ALLOW_DEMO_MODEL","true")
    with TestClient(create_app()) as client:
        assert client.get("/").status_code==200
        assert client.get("/health/ready").json()["status"]=="ready"
        body={"documents":[doc().model_dump()]}
        assert client.post("/v1/classify",json=body).status_code==401
        headers={"X-API-Key":"test-secret-at-least-24-characters"}
        response=client.post("/v1/classify",json=body,headers=headers)
        assert response.status_code==200
        assert response.json()["results"][0]["decision"]=="review"
        assert response.headers["X-Request-ID"]
        bad={"documents":[{"id":"x","text":"secret","extra":"private"}]}
        response=client.post("/v1/classify",json=bad,headers=headers)
        assert response.status_code==422
        assert "private" not in response.text and "secret" not in response.text
        assert client.post("/v1/classify",json={"documents":[]},headers=headers).status_code==422
        assert client.post("/v1/classify",json={"documents":[doc().model_dump()]*2},headers=headers).status_code==422
        assert client.post("/v1/classify",content=b"x"*1000001,headers=headers).status_code==413

def test_demo_blocked(artifact,monkeypatch):
    monkeypatch.setenv("MODEL_PATH",str(artifact));monkeypatch.setenv("API_KEY","test-secret-at-least-24-characters")
    monkeypatch.delenv("ALLOW_DEMO_MODEL",raising=False)
    with pytest.raises(RuntimeError,match="Demo model blocked"):
        with TestClient(create_app()):pass

def test_production_selective_policy(artifact):
    model=Model(artifact);model.demo=False
    model.score=lambda texts:([0.01]*len(texts),[1.0]*len(texts))
    s=ScreeningService(model)
    assert s.predict([doc()])[0]["decision"]=="review"
    assert s.predict([doc(claims={"currency":"INR"},reference={"currency":"INR"})])[0]["decision"]=="no_issue_detected"
    model.score=lambda texts:([0.99]*len(texts),[1.0]*len(texts))
    assert s.predict([doc()])[0]["decision"]=="likely_incorrect"
    model.score=lambda texts:([0.99]*len(texts),[0.1]*len(texts))
    assert s.predict([doc()])[0]["decision"]=="review"


def test_ingestion(tmp_path):
    from app.ingest import ingest
    root=tmp_path/"inbox";root.mkdir()
    (root/"a.txt").write_text("The invoice total matches the verified official billing record")
    (root/"b.eml").write_text("Subject: Check invoice\nContent-Type: text/plain; charset=utf-8\n\nVerify the approved customer identity and official payment record")
    (root/"c.json").write_text(json.dumps(doc().model_dump()))
    (root/"d.pdf").write_bytes(b"pdf")
    out=tmp_path/"out.jsonl";errors=tmp_path/"errors.jsonl"
    assert ingest(root,out,errors)=={"ingested":3,"failed":1}
    rows=[json.loads(x) for x in out.read_text().splitlines()]
    assert [row["source"] for row in rows]==["ticket","email","form"]
    assert len(errors.read_text().splitlines())==1
    with pytest.raises(ValueError):ingest(root,root/"out.jsonl",errors)

def test_malformed_json_model(tmp_path):
    path=tmp_path/"model.json";path.write_text('{"schema_version":99}')
    with pytest.raises(ValueError,match="schema"):Model(path)

def test_api_short_key_rejected(monkeypatch):
    monkeypatch.setenv("API_KEY","short")
    with pytest.raises(RuntimeError,match="API_KEY"):
        with TestClient(create_app()):pass


def test_long_batch_line_is_one_error(artifact,tmp_path):
    source=tmp_path/"long.jsonl";output=tmp_path/"out.jsonl"
    source.write_text("x"*2100000+"\n"+json.dumps(doc().model_dump())+"\n")
    run(source,output,artifact)
    rows=[json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows)==2
    assert rows[0]["error"]=="INPUT_LINE_TOO_LARGE" and rows[1]["line"]==2

def test_thresholds_rejected(artifact):
    with pytest.raises(ValueError,match="thresholds"):ScreeningService(Model(artifact),low=0.9,high=0.1)
