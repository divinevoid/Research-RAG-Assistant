from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate


class LLMJudgeScore(BaseModel):
    relevance: int  # 1–5


JUDGE_PROMPT = PromptTemplate(
    input_variables=["query", "paper"],
    template="""
You are an expert academic reviewer.

Query:
{query}

Paper:
{paper}

Rate the relevance of this paper to the query from 1 to 5.

Return only the number.
"""
)


def llm_judge_score(llm, query, paper_text):
    structured_llm = llm.with_structured_output(LLMJudgeScore)

    chain = JUDGE_PROMPT | structured_llm

    result = chain.invoke({
        "query": query,
        "paper": paper_text[:5000]
    })

    return result.relevance
