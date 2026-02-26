from pydantic import BaseModel, Field
from typing import List

class ResearchIdea(BaseModel):
    title: str = Field(description="A concise, descriptive title for the research idea")
    motivation: str = Field(description="Why this research is needed, citing specific [Paper X].")
    research_question: str = Field(description="The primary question this study aims to answer.")
    novelty: str = Field(description="How this differs from existing work cited in the papers.")
    methodology: str = Field(description="Proposed methods to investigate the research question.")

class ResearchIdeaResponse(BaseModel):
    ideas: List[ResearchIdea] = Field(description="A list of generated research ideas.")