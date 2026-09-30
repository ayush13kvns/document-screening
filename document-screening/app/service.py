import unicodedata
from decimal import Decimal, InvalidOperation
from .model import normalize

SUPPORTED = {"invoice_total", "currency", "customer_id", "account_status"}

def equal_value(key, left, right):
    if key == "invoice_total":
        try:
            a,b = Decimal(left),Decimal(right)
            if not a.is_finite() or not b.is_finite() or a<0 or b<0:
                return None
            return a == b
        except InvalidOperation:
            return None
    # Case sensitive identifiers; currency/status are case insensitive.
    left,right = [unicodedata.normalize("NFKC", v).strip() for v in (left,right)]
    if not left or not right:
        return None
    return left == right if key == "customer_id" else left.casefold()==right.casefold()

class ScreeningService:
    def __init__(self, model, low=0.10, high=0.90, min_coverage=0.60):
        if not 0 <= low < high <= 1 or not 0 <= min_coverage <= 1:
            raise ValueError("Invalid decision thresholds")
        self.model,self.low,self.high,self.min_coverage = model,low,high,min_coverage

    def predict(self, docs):
        probs, coverage = self.model.score([d.text for d in docs])
        results = []
        for doc,p,c in zip(docs,probs,coverage):
            reasons, checks = [], []
            for key,value in doc.claims.items():
                if key not in SUPPORTED:
                    checks.append({"field":key,"status":"unsupported"})
                elif key not in doc.reference:
                    checks.append({"field":key,"status":"missing_reference"})
                else:
                    matched = equal_value(key,value,doc.reference[key])
                    checks.append({"field":key,"status":"invalid_value" if matched is None else "match" if matched else "mismatch"})
            mismatch = any(x["status"]=="mismatch" for x in checks)
            verified = bool(checks) and all(x["status"]=="match" for x in checks)
            short = len(normalize(doc.text).split())<5
            if mismatch:
                decision, basis = "likely_incorrect", "reference_mismatch"
                reasons.append("CLAIM_REFERENCE_MISMATCH")
            elif short or c < self.min_coverage:
                decision,basis = "review", "insufficient_text"
                reasons.append("SHORT_TEXT" if short else "LOW_VOCABULARY_COVERAGE")
            elif self.model.demo:
                decision,basis = "review", "demo_model"
                reasons.append("DEMO_MODEL_NOT_VALIDATED")
            elif p >= self.high:
                decision,basis = "likely_incorrect", "text_model"
                reasons.append("HIGH_TEXT_RISK")
            elif p <= self.low and verified:
                decision,basis = "no_issue_detected", "text_and_reference"
                reasons.append("LOW_TEXT_RISK_AND_SUPPLIED_CLAIMS_MATCH")
            else:
                decision,basis = "review", "uncertain_or_unverified"
                reasons.append("MISSING_OR_INCOMPLETE_REFERENCE" if not verified else "UNCERTAIN_MODEL_SCORE")
            results.append({"id":doc.id,"decision":decision,"basis":basis,
                            "incorrect_probability":round(float(p),6),
                            "vocabulary_coverage":round(float(c),6),
                            "checks":checks,"reasons":reasons,
                            "model_version":self.model.version,"demo_model":self.model.demo})
        return results
