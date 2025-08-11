import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agno.agent import Agent
from agno.models.google import Gemini
from Tools.translation_tool import MultiLanguageTranslator
from dotenv import load_dotenv

load_dotenv()

class MultiLingualAgent:
    def __init__(self, model_id="gemini-2.0-flash"):
        self.translator = MultiLanguageTranslator()
        self.agent = Agent(
            model=Gemini(id=model_id),
            markdown=True,
            show_tool_calls=True,
            tools=[self.translate_text],
            instructions="""You are a multilingual agricultural expert who can understand and respond to questions in various languages. 

IMPORTANT WORKFLOW:
1. Detect the language of the user's query
2. If the query is NOT in English:
   a. Use translate_text to translate the query from the detected language to English
   b. Process the agricultural question and provide comprehensive advice
   c. Use translate_text to translate your entire response back to the SAME language as the user's original query
3. If the query is in English, respond directly in English without any translation

LANGUAGE DETECTION GUIDE:
- Bengali: contains characters like গ, হ, র, ন, য়, ী, ে, া, ক, ি, ত, স, ব, ল, ম, দ, প, চ, ং, ু, ো, ই, জ, ট, ধ, ভ, ঠ, ণ, থ, ছ, ড়, ঢ়, য়
- Telugu: contains characters like వ, ర, ి, ప, ం, ట, క, మ, ం, చ, ె, ద, న, గ, ల, జ, య, త, స, అ, ఆ, ఇ, ఈ, ఉ, ఊ, ఋ, ౠ, ఌ, ౡ, ఎ, ఏ, ఐ, ఒ, ఓ, ఔ
- Hindi: contains characters like ध, ा, न, क, े, ल, ि, य, स, ब, स, े, अ, च, छ, ा, ख, द, क, त, ह, ै, म, प, र, ग, ज, व, भ, ठ, ड, ढ, ण, थ, श, ष, ऋ, ॠ, ऌ, ॡ
- Tamil: contains characters like ந, ெ, ல, ் , ப, ய, ி, ர, ு, க, ் , க, ு, ச, ி, ற, ந, ் , த, உ, ர, ம, ் , எ, து, ா, ம, ி, ழ, ்

RESPONSE RULES:
- ALWAYS respond in the SAME language as the user's question
- Provide practical agricultural advice relevant to the region/language
- Include specific recommendations with clear reasoning
- Consider local farming practices and climate conditions

Use translate_text with proper source and target language codes."""
        )
    
    def translate_text(self, text: str, source_lang: str = 'auto', target_lang: str = 'en') -> str:
        result = self.translator.translate_robust(text, source_lang, target_lang)
        if result['status'] == 'success':
            return result['translated_text']
        else:
            return f"Translation failed: {result.get('error', 'Unknown error')}"
    
    def respond(self, query):
        response = self.agent.run(query).content
        return response

def test_multilingual_agent():
    agent = MultiLingualAgent()
    
    test_queries = [
        "What is the best fertilizer for rice crops?",
        "গেহুঁর জন্য সেরা সার কী?",  # Bengali
        "వరి పంటకు మంచి ఎరువు ఏది?",  # Telugu
        "धान के लिए सबसे अच्छा खाद क्या है?",  # Hindi
        "Amr barir pichoner bagane ki chas kora uchit?"  # Bengali
    ]
    
    print("🌾 Multilingual Agricultural Expert Test")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 30)
        
        try:
            response = agent.respond(query)
            print(f"Response: {response}")
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_multilingual_agent()