from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator

class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128)
    text: str = Field(max_length=30000)
    source: Literal["email", "ticket", "form"] = "form"
    claims: dict[str, str] = Field(default_factory=dict, max_length=30)
    reference: dict[str, str] = Field(default_factory=dict, max_length=30)

    @field_validator("claims", "reference")
    @classmethod
    def bounded_values(cls, value):
        if any(len(k)>64 or len(v)>512 for k,v in value.items()):
            raise ValueError("Reference keys/values exceed limits")
        return value

class Batch(BaseModel):
    documents: list[Document] = Field(min_length=1, max_length=100)

    @field_validator("documents")
    @classmethod
    def unique_ids(cls, docs):
        if len({d.id for d in docs}) != len(docs):
            raise ValueError("Document IDs must be unique within a batch")
        return docs
