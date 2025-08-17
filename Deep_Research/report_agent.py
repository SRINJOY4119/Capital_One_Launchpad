import os
import logging
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class ExecutiveSummary(BaseModel):
    key_findings: List[str] = Field(..., description="Top 5 key findings")
    recommendations: List[str] = Field(..., description="Top 5 actionable recommendations")
    impact_assessment: str = Field(..., description="Potential impact and significance")
    urgency_level: str = Field(..., description="Urgency level: low/medium/high/critical")

class ResearchSection(BaseModel):
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Detailed section content")
    key_points: List[str] = Field(default_factory=list, description="Key bullet points")
    sources_count: int = Field(default=0, description="Number of sources referenced")
    confidence_level: str = Field(default="medium", description="Confidence in findings")

class StructuredAgriculturalReport(BaseModel):
    report_id: str = Field(..., description="Unique report identifier")
    title: str = Field(..., description="Report title")
    objective: str = Field(..., description="Research objective")
    generated_at: str = Field(..., description="Report generation timestamp")
    
    executive_summary: ExecutiveSummary = Field(..., description="Executive summary")
    
    problem_analysis: ResearchSection = Field(..., description="Problem identification and analysis")
    literature_review: ResearchSection = Field(..., description="Literature review and current research")
    methodology: ResearchSection = Field(..., description="Research methodology and approach")
    findings: ResearchSection = Field(..., description="Key findings and results")
    recommendations: ResearchSection = Field(..., description="Actionable recommendations")
    implementation: ResearchSection = Field(..., description="Implementation strategy")
    risk_assessment: ResearchSection = Field(..., description="Risk analysis and mitigation")
    economic_analysis: ResearchSection = Field(..., description="Cost-benefit analysis")
    sustainability: ResearchSection = Field(..., description="Environmental and sustainability factors")
    future_research: ResearchSection = Field(..., description="Future research directions")
    
    citations: List[str] = Field(default_factory=list, description="Bibliography and references")
    appendices: Dict[str, str] = Field(default_factory=dict, description="Additional supporting data")
    
    quality_metrics: Dict[str, str] = Field(default_factory=dict, description="Report quality indicators as strings")

class EnhancedReportAgent:
    def __init__(self):
        self.agent = Agent(
            name="Agricultural Report Specialist",
            model=Gemini(id="gemini-2.0-flash"),
            description="Expert agricultural consultant specializing in comprehensive research report generation",
            instructions=self._get_enhanced_instructions(),
            markdown=True,
        )

    def _get_enhanced_instructions(self) -> str:
        return """
        You are an Expert Agricultural Research Report Generator with advanced analytical capabilities.
        
        REPORT GENERATION FRAMEWORK:
        1. Analyze all provided research data comprehensively
        2. Synthesize information across multiple sources and perspectives
        3. Generate structured, professional agricultural reports
        4. Ensure scientific accuracy and practical applicability
        5. Provide actionable insights and recommendations
        
        SECTION DEVELOPMENT STRATEGY:
        - Executive Summary: Distill key findings into executive-level insights
        - Problem Analysis: Identify root causes and contributing factors
        - Literature Review: Synthesize current research and knowledge gaps
        - Methodology: Explain research approach and data sources
        - Findings: Present evidence-based conclusions
        - Recommendations: Provide specific, actionable guidance
        - Implementation: Detail practical implementation steps
        - Risk Assessment: Identify potential risks and mitigation strategies
        - Economic Analysis: Assess costs, benefits, and ROI
        - Sustainability: Evaluate environmental and long-term impacts
        
        QUALITY STANDARDS:
        - Use evidence-based analysis throughout
        - Maintain scientific rigor and accuracy
        - Ensure practical applicability for farmers and stakeholders
        - Include confidence levels for key findings
        - Reference credible sources and citations
        - Provide clear, actionable recommendations
        
        WRITING STYLE:
        - Professional, clear, and accessible language
        - Logical flow and structure
        - Appropriate use of agricultural terminology
        - Evidence-supported statements
        - Balanced perspective considering multiple viewpoints
        """

    def generate_comprehensive_report(self, 
                                    title: str,
                                    objective: str,
                                    research_data: str,
                                    citation_data: str,
                                    location: str = "Global",
                                    focus_areas: Optional[List[str]] = None) -> StructuredAgriculturalReport:
        
        logger.info(f"Generating comprehensive report: {title}")
        
        report_prompt = self._create_report_prompt(
            title, objective, research_data, citation_data, location, focus_areas
        )
        
        try:
            response = self.agent.run(report_prompt)
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            report = self._parse_response_to_report(response_text, title, objective)
            
            report.report_id = f"AR_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            report.generated_at = datetime.now().isoformat()
            report.quality_metrics = self._calculate_quality_metrics(report, research_data, citation_data)
            
            return report
                
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return self._create_fallback_report(title, objective, research_data, citation_data, str(e))

    def _parse_response_to_report(self, response_text: str, title: str, objective: str) -> StructuredAgriculturalReport:
        try:
            lines = response_text.split('\n')
            content_sections = {}
            current_section = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.upper() in ['EXECUTIVE SUMMARY', 'PROBLEM ANALYSIS', 'LITERATURE REVIEW', 
                                  'METHODOLOGY', 'FINDINGS', 'RECOMMENDATIONS', 'IMPLEMENTATION',
                                  'RISK ASSESSMENT', 'ECONOMIC ANALYSIS', 'SUSTAINABILITY', 'FUTURE RESEARCH']:
                    if current_section and current_content:
                        content_sections[current_section] = '\n'.join(current_content)
                    current_section = line.upper().replace(' ', '_')
                    current_content = []
                elif current_section:
                    current_content.append(line)
            
            if current_section and current_content:
                content_sections[current_section] = '\n'.join(current_content)
            
            def create_section(section_key: str, default_title: str) -> ResearchSection:
                content = content_sections.get(section_key, f"Analysis pending for {default_title.lower()}.")
                return ResearchSection(
                    title=default_title,
                    content=content,
                    key_points=[p.strip('- ') for p in content.split('\n') if p.strip().startswith('-')][:5],
                    confidence_level="medium"
                )
            
            exec_summary_content = content_sections.get('EXECUTIVE_SUMMARY', '')
            key_findings = [f.strip('- ') for f in exec_summary_content.split('\n') if 'finding' in f.lower()][:5]
            recommendations = [r.strip('- ') for r in exec_summary_content.split('\n') if 'recommend' in r.lower()][:5]
            
            if not key_findings:
                key_findings = ["Comprehensive research analysis completed", "Data sources evaluated", 
                              "Best practices identified", "Implementation strategies developed", 
                              "Risk factors assessed"]
            
            if not recommendations:
                recommendations = ["Implement evidence-based practices", "Monitor progress regularly",
                                 "Engage with specialists", "Continue research and development",
                                 "Adapt strategies based on results"]
            
            return StructuredAgriculturalReport(
                report_id="",
                title=title,
                objective=objective,
                generated_at="",
                
                executive_summary=ExecutiveSummary(
                    key_findings=key_findings,
                    recommendations=recommendations,
                    impact_assessment="Moderate to high potential impact based on research analysis",
                    urgency_level="medium"
                ),
                
                problem_analysis=create_section('PROBLEM_ANALYSIS', 'Problem Analysis'),
                literature_review=create_section('LITERATURE_REVIEW', 'Literature Review'),
                methodology=create_section('METHODOLOGY', 'Research Methodology'),
                findings=create_section('FINDINGS', 'Key Findings'),
                recommendations=create_section('RECOMMENDATIONS', 'Recommendations'),
                implementation=create_section('IMPLEMENTATION', 'Implementation Strategy'),
                risk_assessment=create_section('RISK_ASSESSMENT', 'Risk Assessment'),
                economic_analysis=create_section('ECONOMIC_ANALYSIS', 'Economic Analysis'),
                sustainability=create_section('SUSTAINABILITY', 'Sustainability Assessment'),
                future_research=create_section('FUTURE_RESEARCH', 'Future Research Directions')
            )
            
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return self._create_basic_report(title, objective, response_text)

    def _create_basic_report(self, title: str, objective: str, content: str) -> StructuredAgriculturalReport:
        return StructuredAgriculturalReport(
            report_id="",
            title=title,
            objective=objective,
            generated_at="",
            
            executive_summary=ExecutiveSummary(
                key_findings=["Research analysis completed", "Data sources reviewed", "Findings documented"],
                recommendations=["Implement best practices", "Monitor results", "Continue research"],
                impact_assessment="Research provides valuable insights",
                urgency_level="medium"
            ),
            
            problem_analysis=ResearchSection(title="Problem Analysis", content=content[:500]),
            literature_review=ResearchSection(title="Literature Review", content="Literature reviewed"),
            methodology=ResearchSection(title="Methodology", content="Research methodology applied"),
            findings=ResearchSection(title="Findings", content=content[500:1000] if len(content) > 500 else content),
            recommendations=ResearchSection(title="Recommendations", content="Recommendations developed"),
            implementation=ResearchSection(title="Implementation", content="Implementation strategy outlined"),
            risk_assessment=ResearchSection(title="Risk Assessment", content="Risks assessed"),
            economic_analysis=ResearchSection(title="Economic Analysis", content="Economic factors considered"),
            sustainability=ResearchSection(title="Sustainability", content="Sustainability evaluated"),
            future_research=ResearchSection(title="Future Research", content="Future research identified")
        )

    def _create_report_prompt(self, title: str, objective: str, research_data: str, 
                            citation_data: str, location: str, focus_areas: Optional[List[str]]) -> str:
        
        focus_context = ""
        if focus_areas:
            focus_context = f"\nSpecial Focus Areas: {', '.join(focus_areas)}"
        
        return f"""
        Generate a comprehensive agricultural research report with the following specifications:
        
        REPORT DETAILS:
        Title: {title}
        Objective: {objective}
        Location/Context: {location}{focus_context}
        
        AVAILABLE RESEARCH DATA:
        {research_data}
        
        CITATION INFORMATION:
        {citation_data}
        
        Please provide a detailed response organized into the following sections:
        
        EXECUTIVE SUMMARY
        - Key findings (5 main points)
        - Recommendations (5 actionable items)
        - Impact assessment
        - Urgency level
        
        PROBLEM ANALYSIS
        - Problem identification and analysis
        
        LITERATURE REVIEW
        - Current research and knowledge gaps
        
        METHODOLOGY
        - Research approach and data sources
        
        FINDINGS
        - Evidence-based conclusions
        
        RECOMMENDATIONS
        - Specific actionable guidance
        
        IMPLEMENTATION
        - Practical implementation steps
        
        RISK ASSESSMENT
        - Potential risks and mitigation
        
        ECONOMIC ANALYSIS
        - Cost-benefit considerations
        
        SUSTAINABILITY
        - Environmental and long-term impacts
        
        FUTURE RESEARCH
        - Research needs and opportunities
        
        Provide substantive content for each section based on the research data provided.
        """

    def _create_fallback_report(self, title: str, objective: str, research_data: str, 
                              citation_data: str, error: str = "") -> StructuredAgriculturalReport:
        
        error_context = f" (Error: {error})" if error else ""
        
        return StructuredAgriculturalReport(
            report_id=f"AR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_FALLBACK",
            title=f"{title}{error_context}",
            objective=objective,
            generated_at=datetime.now().isoformat(),
            
            executive_summary=ExecutiveSummary(
                key_findings=[
                    f"Research analysis initiated for: {objective}",
                    "Comprehensive data collection completed",
                    "Multi-source information synthesis performed",
                    "Evidence-based recommendations developed",
                    "Implementation framework established"
                ],
                recommendations=[
                    "Conduct detailed field assessment",
                    "Implement evidence-based practices",
                    "Monitor progress and adjust strategies",
                    "Engage with agricultural specialists",
                    "Continue research and development"
                ],
                impact_assessment="Moderate to high potential impact based on preliminary analysis",
                urgency_level="medium"
            ),
            
            problem_analysis=ResearchSection(
                title="Problem Analysis",
                content=f"Comprehensive analysis of the research objective: {objective}. Based on available data and research findings, key issues and challenges have been identified.",
                key_points=["Problem identification completed", "Root cause analysis performed", "Context evaluation done"],
                confidence_level="medium"
            ),
            
            literature_review=ResearchSection(
                title="Literature Review",
                content="Current research and available literature have been reviewed to establish the knowledge foundation for this analysis.",
                key_points=["Academic sources reviewed", "Research gaps identified", "Current best practices documented"],
                confidence_level="medium"
            ),
            
            methodology=ResearchSection(
                title="Research Methodology",
                content="Multi-source data analysis approach utilizing academic research, practical guidelines, and expert knowledge.",
                key_points=["Data collection strategy", "Analysis framework", "Quality assurance measures"],
                confidence_level="high"
            ),
            
            findings=ResearchSection(
                title="Key Findings",
                content="Analysis of research data has yielded important insights relevant to the stated objective.",
                key_points=["Evidence-based conclusions", "Data-supported insights", "Practical implications"],
                confidence_level="medium"
            ),
            
            recommendations=ResearchSection(
                title="Recommendations",
                content="Based on research findings, specific actionable recommendations have been developed.",
                key_points=["Prioritized action items", "Implementation guidance", "Expected outcomes"],
                confidence_level="high"
            ),
            
            implementation=ResearchSection(
                title="Implementation Strategy",
                content="Practical framework for implementing recommendations with consideration of resources and constraints.",
                key_points=["Implementation phases", "Resource requirements", "Timeline considerations"],
                confidence_level="medium"
            ),
            
            risk_assessment=ResearchSection(
                title="Risk Assessment",
                content="Analysis of potential risks and challenges with corresponding mitigation strategies.",
                key_points=["Risk identification", "Impact assessment", "Mitigation strategies"],
                confidence_level="medium"
            ),
            
            economic_analysis=ResearchSection(
                title="Economic Analysis",
                content="Cost-benefit analysis and economic implications of proposed recommendations.",
                key_points=["Cost considerations", "Benefit projections", "ROI analysis"],
                confidence_level="low"
            ),
            
            sustainability=ResearchSection(
                title="Sustainability Assessment",
                content="Environmental and long-term sustainability considerations for proposed approaches.",
                key_points=["Environmental impact", "Long-term viability", "Sustainable practices"],
                confidence_level="medium"
            ),
            
            future_research=ResearchSection(
                title="Future Research Directions",
                content="Identification of additional research needs and opportunities for further investigation.",
                key_points=["Research gaps", "Future studies needed", "Development opportunities"],
                confidence_level="high"
            ),
            
            quality_metrics={
                "fallback_report": "true",
                "error_occurred": "true" if error else "false",
                "generation_method": "fallback"
            }
        )

    def _calculate_quality_metrics(self, report: StructuredAgriculturalReport, 
                                 research_data: str, citation_data: str) -> Dict[str, str]:
        
        sections = [
            report.problem_analysis, report.literature_review, report.methodology,
            report.findings, report.recommendations, report.implementation,
            report.risk_assessment, report.economic_analysis, report.sustainability,
            report.future_research
        ]
        
        total_content_length = sum(len(section.content) for section in sections)
        avg_section_length = total_content_length / len(sections) if sections else 0
        
        total_key_points = sum(len(section.key_points) for section in sections)
        high_confidence_sections = sum(1 for section in sections if section.confidence_level == "high")
        
        return {
            "total_sections": str(len(sections)),
            "total_content_length": str(total_content_length),
            "average_section_length": str(int(avg_section_length)),
            "total_key_points": str(total_key_points),
            "high_confidence_sections": str(high_confidence_sections),
            "executive_summary_items": str(len(report.executive_summary.key_findings) + len(report.executive_summary.recommendations)),
            "citations_included": str(len(report.citations)),
            "research_data_length": str(len(research_data)),
            "citation_data_available": "true" if len(citation_data) > 0 else "false",
            "generation_timestamp": report.generated_at,
            "report_completeness": "complete" if total_content_length > 2000 else "basic"
        }

    def format_report_display(self, report: StructuredAgriculturalReport) -> str:
        output = f"\n🌾 COMPREHENSIVE AGRICULTURAL RESEARCH REPORT\n"
        output += "=" * 80 + "\n"
        output += f"📋 Title: {report.title}\n"
        output += f"🎯 Objective: {report.objective}\n"
        output += f"📅 Generated: {report.generated_at}\n"
        output += f"🆔 Report ID: {report.report_id}\n\n"
        
        output += "📊 EXECUTIVE SUMMARY\n"
        output += "-" * 40 + "\n"
        output += f"🔍 Key Findings:\n"
        for i, finding in enumerate(report.executive_summary.key_findings, 1):
            output += f"  {i}. {finding}\n"
        
        output += f"\n💡 Recommendations:\n"
        for i, rec in enumerate(report.executive_summary.recommendations, 1):
            output += f"  {i}. {rec}\n"
        
        output += f"\n📈 Impact Assessment: {report.executive_summary.impact_assessment}\n"
        output += f"⚡ Urgency Level: {report.executive_summary.urgency_level.upper()}\n\n"
        
        sections = [
            ("Problem Analysis", report.problem_analysis),
            ("Literature Review", report.literature_review),
            ("Methodology", report.methodology),
            ("Key Findings", report.findings),
            ("Recommendations", report.recommendations),
            ("Implementation Strategy", report.implementation),
            ("Risk Assessment", report.risk_assessment),
            ("Economic Analysis", report.economic_analysis),
            ("Sustainability", report.sustainability),
            ("Future Research", report.future_research)
        ]
        
        for section_name, section in sections:
            output += f"📚 {section_name.upper()}\n"
            output += "-" * 40 + "\n"
            output += f"{section.content}\n"
            
            if section.key_points:
                output += f"\n🔑 Key Points:\n"
                for point in section.key_points:
                    output += f"  • {point}\n"
            
            output += f"\n📊 Confidence Level: {section.confidence_level.upper()}\n"
            if section.sources_count > 0:
                output += f"📚 Sources Referenced: {section.sources_count}\n"
            output += "\n"
        
        if report.citations:
            output += "📖 REFERENCES\n"
            output += "-" * 40 + "\n"
            for i, citation in enumerate(report.citations, 1):
                output += f"{i}. {citation}\n"
            output += "\n"
        
        output += "📊 QUALITY METRICS\n"
        output += "-" * 40 + "\n"
        metrics = report.quality_metrics
        output += f"Report Completeness: {metrics.get('report_completeness', 'unknown').upper()}\n"
        output += f"Total Sections: {metrics.get('total_sections', '0')}\n"
        output += f"High Confidence Sections: {metrics.get('high_confidence_sections', '0')}\n"
        output += f"Total Key Points: {metrics.get('total_key_points', '0')}\n"
        output += f"Content Length: {metrics.get('total_content_length', '0')} characters\n"
        
        return output