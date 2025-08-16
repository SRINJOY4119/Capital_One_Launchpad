from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class AnswerGradingResponse(BaseModel):
    feedback: str
    decision: bool

class AnswerGraderAgent:
    def __init__(self):
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            instructions="""
You are an expert answer grading assistant for educational assessments. Your task is to evaluate the quality of answers provided by students based on the following rubric:

RUBRIC CRITERIA:
- Accuracy: Is the answer factually correct and free from errors?
- Completeness: Does the answer address all parts of the question thoroughly?
- Clarity: Is the answer clearly explained, well-structured, and easy to understand?
- Relevance: Is the answer focused on the question and avoids unnecessary information?

INSTRUCTIONS:
- Carefully read the question and the student's answer.
- Assess the answer against each rubric criterion.
- Provide detailed, constructive feedback highlighting strengths and areas for improvement.
- Clearly state whether the answer meets the expected standard (decision: True/False).
- Use a supportive and professional tone.
- Output should be an AnswerGradingResponse object with 'feedback' and 'decision'.
- Do not mention tool calling or internal implementation details.
""",
            response_model=AnswerGradingResponse,
        )

    def grade(self, question: str, answer: str) -> AnswerGradingResponse:
        prompt = f"Question: {question}\nAnswer: {answer}\nEvaluate the answer as per the rubric and provide feedback and decision."
        return self.agent.run(prompt).content

if __name__ == "__main__":
    grader = AnswerGraderAgent()
    question = "Explain the process of photosynthesis."
    answer = "Photosynthesis is the process by which plants make their food using sunlight, water, and carbon dioxide."
    result = grader.grade(question, answer)
    print(result)

