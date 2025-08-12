from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ContextRelevance
from langchain_groq import ChatGroq
import pandas as pd
import asyncio
import os
from dotenv import load_dotenv

# Compatibility wrapper for Ragas
class ChatGroqWrapper:
    def __init__(self, chatgroq):
        self.chatgroq = chatgroq

    def generate_text(self, prompt: str, **kwargs):
        """Sync text generation"""
        response = self.chatgroq.invoke(prompt, **kwargs)
        return response.content if hasattr(response, "content") else str(response)

    async def agenerate_text(self, prompt: str, **kwargs):
        """Async text generation"""
        response = await self.chatgroq.ainvoke(prompt, **kwargs)
        return response.content if hasattr(response, "content") else str(response)

    def __getattr__(self, name):
        return getattr(self.chatgroq, name)

async def evaluate_agricultural_advice(fertilizer_data, crop_data, recommendation):
    """Evaluate if agricultural recommendation matches known good practices"""
    
    # Load environment variables and set up Groq LLM
    load_dotenv()
    chatgroq = ChatGroq(
        model="llama3-70b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Wrap it so Ragas can use it
    llm = ChatGroqWrapper(chatgroq)
    
    # Create context from our agricultural datasets
    contexts = [
        f"""Crop requirements:
        {crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']].to_string()}""",
        
        f"""Fertilizer applications:
        {fertilizer_data[['Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous', 'Fertilizer Name']].to_string()}"""
    ]
    
    # Initialize scorer with Groq LLM
    context_scorer = ContextRelevance(llm=llm)
    
    sample = SingleTurnSample(
        user_input="What are the appropriate agricultural practices for these conditions?",
        retrieved_contexts=contexts,
        assistant_response=recommendation
    )
    
    score = await context_scorer.single_turn_ascore(sample)
    return score

async def main():
    # Load our agricultural datasets
    crop_data = pd.read_csv(r'../Dataset/crop_recommendation.csv')
    fertilizer_data = pd.read_csv(r'../Dataset/Fertilizer_recommendation.csv')
    
    # Test recommendation based on actual data
    valid_recommendation = """For cotton cultivation in black soil:
    - Maintain temperature between 21-30°C
    - Apply NPK in ratio 100:50:50 kg/ha
    - Keep soil pH between 6.0-7.5
    - Ensure proper drainage
    - Use split application of nitrogen"""
    
    # Test invalid recommendation
    invalid_recommendation = """For cotton cultivation:
    - Keep temperature at 45°C
    - Apply NPK 1000:1000:1000 kg/ha
    - No need to check pH
    - Maintain waterlogged conditions"""
    
    print("Testing valid agricultural recommendation:")
    valid_score = await evaluate_agricultural_advice(
        fertilizer_data, 
        crop_data, 
        valid_recommendation
    )
    print(f"Score: {valid_score:.2f}")
    print(f"Assessment: {'Factual' if valid_score > 0.7 else 'Potential hallucination'}")
    
    print("\nTesting invalid agricultural recommendation:")
    invalid_score = await evaluate_agricultural_advice(
        fertilizer_data, 
        crop_data, 
        invalid_recommendation
    )
    print(f"Score: {invalid_score:.2f}")
    print(f"Assessment: {'Factual' if invalid_score > 0.7 else 'Potential hallucination'}")

if __name__ == "__main__":
    asyncio.run(main())
