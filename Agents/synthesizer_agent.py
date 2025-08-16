from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv

load_dotenv()

class SynthesizerAgent:
    def __init__(self):
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            instructions="""
You are a synthesis and summarization expert for agricultural AI. Your job is to take multiple responses from different agents (RAG, model, or tool outputs) and refactor them into a single, clear, actionable, and well-structured answer for the user.

INSTRUCTIONS:
- Carefully read and analyze each response in the provided list.
- Identify key points, insights, and recommendations from each response.
- Remove redundancy, resolve contradictions, and synthesize information into a coherent summary.
- Present the final result in a logical, readable format with clear sections, bullet points, and actionable advice.
- Use a professional, helpful, and concise tone.
- Do not mention tool calling or internal implementation details.
- The output should be suitable for direct presentation to the user.
"""
        )

    def synthesize(self, responses: list[str]) -> str:
        prompt = (
            "Given the following responses from multiple agents, synthesize and refactor them into a single, clear, actionable, and well-structured answer for the user.\n\n"
            "Responses:\n"
        )
        for i, resp in enumerate(responses, 1):
            prompt += f"Response {i}:\n{resp}\n\n"
        prompt += "Provide the final synthesized answer below:\n"
        return self.agent.run(prompt).content

if __name__ == "__main__":
    responses = [
        "The weather forecast for Nashik indicates moderate rainfall and temperatures suitable for Kharif crops.",
        "Recommended crops for Nashik in Kharif season are rice, soybean, and maize due to local soil and climate conditions.",
        "Market prices for rice and soybean are favorable, and risk levels are moderate for these crops."
    ]
    synthesizer = SynthesizerAgent()
    final_answer = synthesizer.synthesize(responses)
    print(final_answer)
