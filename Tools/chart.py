from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.python import PythonTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import re

load_dotenv()

class RealDataAgricultureChartAgent:
    def __init__(self):
        """Initialize the agriculture-focused chart generation agent with real data capabilities"""
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            tools=[
                PythonTools(),
                GoogleSearchTools(),
                DuckDuckGoTools(), 
                YFinanceTools()
            ],
            show_tool_calls=True,
            markdown=True,
            instructions="""
You are an agricultural data visualization expert with access to real-time data. Your role is to automatically generate relevant charts using REAL DATA for farming and agriculture queries.

DATA SOURCES TO USE:
1. **YFinance**: For commodity prices (wheat, corn, soybeans, cotton, etc.)
   - Use symbols like: ZW=F (wheat), ZC=F (corn), ZS=F (soybeans), CT=F (cotton)
   - Get historical price data for trends

2. **Google Search**: For current market prices, weather data, agricultural reports
   - Search for: "current wheat prices 2024", "fertilizer prices today", "crop yield statistics"
   - Look for USDA reports, agricultural market data

3. **DuckDuckGo Search**: Alternative search for agricultural data, weather information
   - Search for: "agricultural commodity prices", "farming costs 2024", "crop insurance rates"

REAL DATA RETRIEVAL PROCESS:
1. **First**: Always try to get real data using the appropriate tool
2. **YFinance**: For commodity futures and agricultural stock prices
3. **Search Tools**: For current market data, costs, yields, weather
4. **Fallback**: Only use sample data if real data is unavailable

REQUIRED LIBRARIES TO USE:
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
import yfinance as yf
plt.style.use('seaborn-v0_8')
```

CHART GENERATION WORKFLOW:
1. **Analyze Query** → Determine what real data is needed
2. **Fetch Real Data** → Use appropriate tools to get current data
3. **Process Data** → Clean and prepare for visualization
4. **Generate Chart** → Create relevant chart with real data
5. **Provide Insights** → Give actionable advice based on real data

COMMODITY SYMBOLS FOR YFINANCE:
- Wheat: ZW=F
- Corn: ZC=F  
- Soybeans: ZS=F
- Cotton: CT=F
- Sugar: SB=F
- Coffee: KC=F
- Rice: ZR=F
- Oats: ZO=F

SEARCH QUERY EXAMPLES:
- "USDA corn prices December 2024"
- "fertilizer costs per acre 2024"
- "agricultural machinery prices current"
- "crop yield statistics by state 2024"
- "weather forecast agriculture impact"

CHART REQUIREMENTS:
- Use REAL data whenever possible
- Create professional agricultural visualizations
- Include data sources in chart titles/annotations
- Use appropriate colors (greens, browns, earth tones)
- Save charts with descriptive names including date
- Always show data collection timestamp
- Provide insights based on REAL market conditions

Remember: ALWAYS attempt to fetch real data first before using any sample data!
"""
        )
    
    def detect_chart_trigger(self, query):
        """Detect if query should trigger automatic chart generation"""
        price_keywords = ['price', 'cost', 'expensive', 'cheap', 'rates', 'market']
        comparison_keywords = ['better', 'best', 'compare', 'vs', 'versus', 'which', 'difference']
        trend_keywords = ['trend', 'changing', 'increase', 'decrease', 'forecast', 'prediction']
        planning_keywords = ['when', 'timing', 'schedule', 'plant', 'harvest', 'irrigate']
        roi_keywords = ['profit', 'profitable', 'return', 'roi', 'investment', 'income']
        
        query_lower = query.lower()
        
        triggers = []
        if any(keyword in query_lower for keyword in price_keywords):
            triggers.append('price')
        if any(keyword in query_lower for keyword in comparison_keywords):
            triggers.append('comparison') 
        if any(keyword in query_lower for keyword in trend_keywords):
            triggers.append('trend')
        if any(keyword in query_lower for keyword in planning_keywords):
            triggers.append('planning')
        if any(keyword in query_lower for keyword in roi_keywords):
            triggers.append('roi')
            
        return triggers
    
    def identify_data_sources(self, query):
        """Identify what real data sources should be used for the query"""
        query_lower = query.lower()
        data_sources = []
        
        # Commodity futures data
        commodities = ['wheat', 'corn', 'soybean', 'cotton', 'sugar', 'coffee', 'rice', 'oats']
        if any(commodity in query_lower for commodity in commodities):
            data_sources.append('yfinance')
        
        # Current market data, costs, reports
        market_keywords = ['current', 'today', 'latest', 'recent', 'market', 'usda', 'report']
        if any(keyword in query_lower for keyword in market_keywords):
            data_sources.append('search')
        
        # Weather and planning data
        weather_keywords = ['weather', 'climate', 'rainfall', 'temperature', 'season']
        if any(keyword in query_lower for keyword in weather_keywords):
            data_sources.append('search')
        
        return data_sources
    
    def generate_chart_response(self, query):
        """Generate appropriate chart based on query type using real data"""
        triggers = self.detect_chart_trigger(query)
        data_sources = self.identify_data_sources(query)
        
        if not triggers:
            # Regular response without chart
            return self.agent.print_response(query)
        
        # Create enhanced chart generation prompt with real data instructions
        chart_prompt = f"""
Agricultural Query: "{query}"

Detected chart needs: {', '.join(triggers)}
Recommended data sources: {', '.join(data_sources) if data_sources else 'search for current data'}

STEP-BY-STEP REAL DATA CHART GENERATION:

1. **FETCH REAL DATA FIRST**:
   - If query mentions commodities (wheat, corn, etc.), use YFinanceTools to get current and historical prices
   - Use GoogleSearch or DuckDuckGo to find current agricultural market data, costs, yields
   - Search for USDA reports, agricultural statistics, current market conditions

2. **DATA COLLECTION EXAMPLES**:
   ```python
   # For commodity prices
   import yfinance as yf
   wheat_data = yf.download('ZW=F', period='1y')  # Wheat futures
   corn_data = yf.download('ZC=F', period='6mo')  # Corn futures
   rice_data = yf.download('ZR=F', period='1y')   # Rice futures
   ```
   
3. **SEARCH FOR CURRENT DATA**:
   - Search: "current rice prices December 2024"
   - Search: "fertilizer costs per acre 2024" 
   - Search: "USDA crop report latest"
   - Search: "agricultural equipment prices current"
   - Search: "rice market growth statistics 2024"

4. **GENERATE CHART WITH REAL DATA**:
   - Process the fetched real data
   - Create appropriate visualization (line/bar/pie chart)
   - Include data source and timestamp in chart
   - Use professional agricultural styling

5. **PROVIDE REAL MARKET INSIGHTS**:
   - Analyze current market conditions
   - Give actionable advice based on real data trends
   - Mention data collection date and sources

**IMPORTANT**: Always attempt to fetch real data before creating any chart. If real data is unavailable, clearly state this and explain what data sources were attempted.

Generate the Python code to fetch real data and create this agricultural visualization now.
"""
        
        return self.agent.print_response(chart_prompt)

def main():
    """Main function to demonstrate the real data chart generation system"""
    chart_agent = RealDataAgricultureChartAgent()
    
    # Example agricultural queries that will use real data
    sample_queries = [
        "Show me current wheat prices and 6-month trend",
        "What are corn futures doing this month?", 
        "Compare soybean vs wheat price performance",
        "Current fertilizer costs per acre in 2024",
        "Cotton price forecast based on recent trends",
        "Is organic farming profitable with current market prices?",
        "Show me agricultural commodity price trends",
        "Current crop insurance rates vs historical data",
        "Weather impact on corn prices this season",
        "USDA crop report data visualization"
    ]
    
    print("🌾 Real Data Agricultural Chart Generation System")
    print("="*60)
    print("📊 Using: YFinance + GoogleSearch + DuckDuckGo for real data")
    print("="*60)
    
    while True:
        print("\nSample queries (using real data):")
        for i, query in enumerate(sample_queries, 1):
            print(f"{i}. {query}")
        
        print("\nEnter your agricultural query (or 'quit' to exit):")
        user_query = input("Query: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using the Real Data Agricultural Chart System! 🚜")
            break
            
        if not user_query:
            continue
            
        # Check if it's a sample query number
        if user_query.isdigit() and 1 <= int(user_query) <= len(sample_queries):
            user_query = sample_queries[int(user_query) - 1]
            print(f"Selected query: {user_query}")
        
        print(f"\n🔍 Processing with real data: {user_query}")
        print("-" * 50)
        
        # Generate response with real data fetching
        chart_agent.generate_chart_response(user_query)
        
        print("\n" + "="*60)

if __name__ == "__main__":
    main()