# So now we are creating a graph
# and the first thing you create is a state

import os

# 1) typed DICT (Most common aaproach)

from typing import TypedDict


class State(TypedDict):
    topic: str
    summary: str
    score: str


# 2 pydantic approach
# It is good at data validation and type checking at run time

from pydantic import BaseModel, field_validator


class State(BaseModel):
    topic: str
    score: int
    summary: str = ""



    @field_validator("score")
    @classmethod
    def score_positive(cls, v):
        if v < 0:
            raise ValueError("Score must be positive.")
        
        return v
