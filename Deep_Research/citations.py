import sys
import os
import requests
import json
from typing import List
from urllib.parse import urlparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'Tools'))

from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from agno.tools.tavily import TavilyTools
from Tools.web_scrapper import scrape_agri_prices, scrape_policy_updates, scrape_links

load_dotenv()

class AgriculturalCitationAgent:
    def __init__(self):
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash-exp"),
            tools=[TavilyTools(), scrape_agri_prices, scrape_policy_updates, scrape_links],
            instructions=self._get_instructions(),
            markdown=True,
        )
        
        # Trusted agricultural domains for validation
        self.trusted_domains = [
            'nature.com', 'sciencedirect.com', 'springer.com', 'wiley.com',
            'tandfonline.com', 'cambridge.org', 'oxford.com', 'plos.org',
            'usda.gov', 'fao.org', 'who.int', 'cgiar.org', 'worldbank.org',
            'nih.gov', 'pubmed.ncbi.nlm.nih.gov', 'researchgate.net',
            'frontiersin.org', 'mdpi.com', 'elsevier.com', 'jstor.org',
            'acsess.onlinelibrary.wiley.com', 'journals.ashs.org',
            'link.springer.com', 'onlinelibrary.wiley.com', 'journals.plos.org'
        ]

    def _get_instructions(self) -> str:
        return """
You are an agricultural research citation specialist. Your job is to find high-quality, credible citations with valid links for agricultural topics.

CITATION SEARCH & SCRAPING FRAMEWORK:
- Use TavilyTools for comprehensive web search to find academic papers, research reports, and credible sources
- Use scrape_links to extract and gather URLs from agricultural research websites, journals, and institutional pages
- Use scrape_policy_updates to find recent agricultural policy documents and government publications that can serve as citations
- Use scrape_agri_prices when relevant to find market research and economic studies related to agricultural topics
- Combine search results with web scraping to ensure comprehensive citation coverage

SEARCH & SCRAPING STRATEGY:
- First use TavilyTools to identify relevant agricultural research websites and journal pages
- Then use scrape_links to extract citation URLs from those pages
- Use scrape_policy_updates for government and policy-related agricultural research
- Focus on peer-reviewed journals, government publications, and institutional research
- Target sources from: Agricultural Systems, Journal of Agricultural Science, USDA, FAO, university extensions
- Prioritize recent publications (last 10 years) unless specified otherwise

WEB SCRAPING FOR CITATIONS:
- Use scrape_links to extract academic paper URLs from journal websites and research portals
- Scrape university extension websites and research institution pages for publications
- Extract DOIs and direct links to research papers from scraped pages
- Gather conference proceeding links and institutional report URLs
- Validate all scraped URLs for accessibility and relevance

OUTPUT REQUIREMENTS:
- Provide complete bibliographic information for each source found via search and scraping
- Include working URLs or DOIs extracted from scraped pages
- Format citations in APA style
- Rate relevance to the topic (1-10 scale)  
- Categorize sources by type (Journal Article, Report, Extension Publication, etc.)
- Present results with:
  * Title and authors
  * Publication details
  * Working URL/DOI (validated from scraping)
  * APA citation format
  * Relevance score
  * Source type

QUALITY ASSURANCE:
- Verify all scraped URLs are accessible before including
- Cross-reference search results with scraped content
- Prioritize peer-reviewed and authoritative sources found through scraping
- Ensure direct relevance to the agricultural topic
- Focus on English language sources unless specified otherwise

Always combine web search with targeted scraping to provide comprehensive, accessible citations.
"""

    def validate_url(self, url: str) -> bool:
        """Validate if a URL is accessible and from trusted source."""
        try:
            if not url or not url.startswith(('http://', 'https://', 'doi:')):
                return False
                
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower().replace('www.', '')
            
            # Check if from trusted domain
            is_trusted = any(trusted in domain for trusted in self.trusted_domains)
            if not is_trusted:
                return False
            
            # Test URL accessibility with timeout
            response = requests.head(url, timeout=8, allow_redirects=True)
            return response.status_code == 200
            
        except Exception:
            return False

    def extract_citations_from_response(self, response_text: str) -> List[dict]:
        """Extract and validate citations from agent response."""
        citations = []
        
        try:
            # Look for structured information in the response
            lines = response_text.split('\n')
            current_citation = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_citation and 'title' in current_citation:
                        citations.append(current_citation.copy())
                        current_citation = {}
                    continue
                
                # Extract various citation components
                if line.startswith('Title:') or line.startswith('**Title'):
                    current_citation['title'] = line.split(':', 1)[-1].strip().replace('**', '')
                elif line.startswith('Authors:') or line.startswith('**Authors'):
                    current_citation['authors'] = line.split(':', 1)[-1].strip().replace('**', '')
                elif line.startswith('Journal:') or line.startswith('**Journal') or line.startswith('Source:'):
                    current_citation['journal'] = line.split(':', 1)[-1].strip().replace('**', '')
                elif line.startswith('Year:') or line.startswith('**Year'):
                    current_citation['year'] = line.split(':', 1)[-1].strip().replace('**', '')
                elif line.startswith('URL:') or line.startswith('**URL') or line.startswith('DOI:') or 'http' in line:
                    url = line.split(':', 1)[-1].strip().replace('**', '') if ':' in line else line
                    if url.startswith('http') or url.startswith('doi:'):
                        current_citation['url'] = url
                elif line.startswith('Type:') or line.startswith('**Type'):
                    current_citation['type'] = line.split(':', 1)[-1].strip().replace('**', '')
                elif line.startswith('Relevance:') or line.startswith('**Relevance'):
                    try:
                        score = line.split(':', 1)[-1].strip().replace('**', '')
                        current_citation['relevance'] = int(score.split('/')[0])
                    except:
                        current_citation['relevance'] = 8
            
            # Add the last citation if exists
            if current_citation and 'title' in current_citation:
                citations.append(current_citation)
                
        except Exception as e:
            print(f"⚠️ Error extracting citations: {e}")
            
        return citations

    def find_citations(self, topic: str, num_citations: int = 10) -> str:
        """Find and return valid agricultural citations for the given topic."""
        
        prompt = f"""
Find {num_citations} high-quality agricultural citations for the topic: "{topic}"

SEARCH & SCRAPING STRATEGY:
1. First use TavilyTools to search for relevant agricultural research websites, journal pages, and institutional sources
2. Use scrape_links to extract citation URLs from the found websites and research portals  
3. Use scrape_policy_updates if the topic relates to agricultural policies or government research
4. Use scrape_agri_prices if the topic involves agricultural economics or market research

Please search for and scrape:
1. Peer-reviewed journal articles from agricultural journals
2. Government research reports (USDA, FAO, etc.)
3. University extension publications  
4. Conference papers and proceedings
5. Institutional research studies

For each citation found through search and scraping, provide:
- Title: [Full title]
- Authors: [Author names]
- Journal: [Journal/Source name]  
- Year: [Publication year]
- URL: [Working URL or DOI extracted from scraping]
- Type: [Source type]
- Relevance: [Score 1-10]

Use your web scraping tools to extract direct links to research papers and validate their accessibility. Focus on recent publications with working URLs.
"""
        
        print("🔍 Searching for agricultural citations...")
        response = self.agent.run(prompt)
        response_text = response.content
        
        print("🔗 Extracting and validating citations...")
        citations_data = self.extract_citations_from_response(response_text)
        
        # Validate URLs and format output
        valid_citations = []
        for citation in citations_data:
            if 'url' in citation and self.validate_url(citation['url']):
                valid_citations.append(citation)
            elif 'url' in citation:
                print(f"⚠️ Invalid URL for: {citation.get('title', 'Unknown')}")
        
        # Format final output
        return self._format_citations_output(valid_citations, topic, response_text)

    def _format_citations_output(self, citations: List[dict], topic: str, raw_response: str) -> str:
        """Format citations for final output."""
        
        output = f"\n📚 Agricultural Citations for: '{topic}'\n"
        output += "=" * 60 + "\n"
        
        if not citations:
            output += "\n❌ No valid citations with accessible URLs found.\n"
            output += "\n📄 Raw search results:\n"
            output += "-" * 40 + "\n"
            output += raw_response[:1000] + "..." if len(raw_response) > 1000 else raw_response
            return output
        
        for i, citation in enumerate(citations, 1):
            title = citation.get('title', 'Unknown Title')
            authors = citation.get('authors', 'Unknown Authors')
            journal = citation.get('journal', 'Unknown Source')
            year = citation.get('year', 'Unknown Year')
            url = citation.get('url', '')
            source_type = citation.get('type', 'Unknown Type')
            relevance = citation.get('relevance', 'N/A')
            
            # Create APA citation format
            apa_citation = f"{authors} ({year}). {title}. {journal}."
            
            output += f"\n{i}. {apa_citation}\n"
            output += f"   🔗 URL: {url}\n"
            output += f"   📊 Relevance: {relevance}/10\n"
            output += f"   📑 Type: {source_type}\n"
            output += "-" * 50 + "\n"
        
        output += f"\n✅ Total Valid Citations: {len(citations)}\n"
        output += f"💡 All URLs have been validated for accessibility.\n"
        
        return output

    def get_citations(self, topic: str, num_citations: int = 10) -> str:
        """Main method to get citations for a topic."""
        return self.find_citations(topic, num_citations)

def main():
    print("🌾 Agricultural Citation Agent")
    print("=" * 40)
    
    topic = input("\n📝 Enter agricultural topic: ").strip()
    
    if not topic:
        print("❌ Please provide a topic.")
        return
    
    try:
        num_citations = int(input("🔢 Number of citations (default 10): ").strip() or "10")
    except ValueError:
        num_citations = 10
    
    print(f"\n🚀 Finding citations for: '{topic}'")
    print("⏳ Please wait...\n")
    
    agent = AgriculturalCitationAgent()
    result = agent.get_citations(topic, num_citations)
    print(result)
    
    # Save option
    save = input("\n💾 Save results to file? (y/n): ").strip().lower()
    if save == 'y':
        filename = f"citations_{topic.replace(' ', '_')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"✅ Saved to: {filename}")

if __name__ == "__main__":
    main()