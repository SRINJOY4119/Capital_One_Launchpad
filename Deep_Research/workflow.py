import os
import sys
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import logging
from datetime import datetime

# PDF generation imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import textwrap

from .planner_agent import AgriculturalPlanningAgent
from .SubsearchAgent import EnhancedSubsearchAgent
from .citations_agent import EnhancedCitationAgent
from .report_agent import EnhancedReportAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchState:
    title: str = ""
    objective: str = ""
    location: str = "Global"
    focus_areas: List[str] = None
    
    plan: Any = None
    tasks: List[Any] = None
    
    research_results: Dict[str, Any] = None
    citation_results: Dict[str, Any] = None
    
    final_report: Any = None
    
    current_phase: str = "initialized"
    errors: List[str] = None
    execution_log: List[str] = None
    
    def __post_init__(self):
        if self.focus_areas is None:
            self.focus_areas = []
        if self.research_results is None:
            self.research_results = {}
        if self.citation_results is None:
            self.citation_results = {}
        if self.errors is None:
            self.errors = []
        if self.execution_log is None:
            self.execution_log = []

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        if 'CustomTitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Title'],
                fontSize=18,
                textColor=colors.darkgreen,
                alignment=TA_CENTER,
                spaceAfter=20
            ))
        if 'SectionHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading2'],
                fontSize=14,
                textColor=colors.darkblue,
                spaceAfter=12,
                spaceBefore=16
            ))
        if 'SubHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SubHeader',
                parent=self.styles['Heading3'],
                fontSize=12,
                textColor=colors.darkred,
                spaceAfter=8,
                spaceBefore=12
            ))
        if 'CustomBodyText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomBodyText',
                parent=self.styles['Normal'],
                fontSize=10,
                alignment=TA_JUSTIFY,
                spaceAfter=6
            ))
        if 'Citation' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='Citation',
                parent=self.styles['Normal'],
                fontSize=9,
                leftIndent=20,
                spaceAfter=4
            ))
    
    def _sanitize_text(self, text: str) -> str:
        if not text:
            return ""
        text = str(text)
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        return text
    
    def generate_pdf_report(self, results: Dict[str, Any], filename: str):
        os.makedirs('Reports', exist_ok=True)
        
        doc = SimpleDocTemplate(filename, pagesize=A4, 
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        story = []
        
        story.extend(self._create_title_page(results))
        story.append(PageBreak())
        
        story.extend(self._create_executive_summary(results))
        story.append(PageBreak())
        
        story.extend(self._create_research_details(results))
        
        if results.get('final_report'):
            story.extend(self._create_main_content(results['final_report']))
        
        if results.get('citation_results'):
            story.extend(self._create_citations_section(results['citation_results']))
        
        story.extend(self._create_appendix(results))
        
        doc.build(story)
        logger.info(f"PDF report generated: {filename}")
    
    def _create_title_page(self, results: Dict[str, Any]) -> List:
        elements = []
        
        title = self._sanitize_text(results.get('title', 'Agricultural Research Report'))
        elements.append(Paragraph(title, self.styles['CustomTitle']))
        elements.append(Spacer(1, 20))
        
        objective = self._sanitize_text(results.get('objective', ''))
        if objective:
            elements.append(Paragraph(f"<b>Research Objective:</b><br/>{objective}", 
                                    self.styles['Normal']))
            elements.append(Spacer(1, 15))
        
        metadata = [
            ['Location:', self._sanitize_text(results.get('location', 'Global'))],
            ['Generated:', datetime.now().strftime('%B %d, %Y at %H:%M')],
            ['Research Type:', 'Deep Agricultural Analysis'],
        ]
        
        if results.get('focus_areas'):
            focus_text = ', '.join(results['focus_areas'])
            metadata.append(['Focus Areas:', self._sanitize_text(focus_text)])
        
        table = Table(metadata, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 30))
        
        quality = results.get('quality_validation', {})
        if quality:
            quality_text = f"""
            <b>Research Quality Assessment</b><br/>
            Overall Quality: {quality.get('overall_quality', 'Unknown').upper()}<br/>
            Quality Score: {quality.get('quality_score', 0)}/100<br/>
            Citations Found: {quality.get('citations_found', 0)}<br/>
            Sources Analyzed: {quality.get('sources_found', 0)}
            """
            elements.append(Paragraph(quality_text, self.styles['Normal']))
        
        return elements
    
    def _create_executive_summary(self, results: Dict[str, Any]) -> List:
        elements = []
        
        elements.append(Paragraph("EXECUTIVE SUMMARY", self.styles['SectionHeader']))
        
        exec_summary = results.get('execution_summary', {})
        phases_completed = exec_summary.get('phases_completed', 0)
        errors_count = exec_summary.get('errors_encountered', 0)
        
        summary_text = f"""
        This comprehensive agricultural research study was conducted using an automated 
        deep research workflow. The research successfully completed {phases_completed} out of 5 
        planned phases with {errors_count} errors encountered during execution.
        """
        
        elements.append(Paragraph(self._sanitize_text(summary_text.strip()), 
                                self.styles['BodyText']))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("Key Research Components", self.styles['SubHeader']))
        
        components = []
        
        planning = results.get('planning_results', {})
        if planning.get('plan_generated'):
            components.append(f"Research plan with {planning.get('tasks_created', 0)} tasks generated")
            components.append(f"{planning.get('agent_assignments', 0)} specialist agents assigned")
        
        research = results.get('research_results', {})
        if research and research.get('success'):
            components.append(f"{research.get('successful_searches', 0)} successful research queries")
            components.append(f"{research.get('total_sources_found', 0)} sources analyzed")
        
        citations = results.get('citation_results', {})
        if citations and citations.get('success'):
            components.append(f"{citations.get('valid_count', 0)} validated citations collected")
        
        if results.get('final_report'):
            components.append("Comprehensive research report generated")
        
        for component in components:
            elements.append(Paragraph(self._sanitize_text(component), self.styles['CustomBodyText']))
        
        return elements
    
    def _create_research_details(self, results: Dict[str, Any]) -> List:
        elements = []
        
        elements.append(Paragraph("RESEARCH METHODOLOGY", self.styles['SectionHeader']))
        
        methodology_text = """
        This research employed a multi-phase automated approach combining strategic planning,
        parallel research execution, comprehensive citation gathering, and quality validation.
        The methodology ensures thorough coverage of the research topic while maintaining
        high standards of academic rigor.
        """
        
        elements.append(Paragraph(self._sanitize_text(methodology_text.strip()), 
                                self.styles['CustomBodyText']))
        
        elements.append(Paragraph("Research Phases", self.styles['SubHeader']))
        
        phase_descriptions = [
            ("Planning Phase", "Strategic decomposition of research objectives into actionable tasks"),
            ("Research Execution", "Multi-threaded information gathering from diverse sources"),
            ("Citation Gathering", "Academic source validation and bibliography compilation"),
            ("Report Generation", "Synthesis of findings into comprehensive documentation"),
            ("Quality Validation", "Assessment of research completeness and reliability")
        ]
        
        for phase_name, description in phase_descriptions:
            phase_text = f"<b>{phase_name}:</b> {description}"
            elements.append(Paragraph(self._sanitize_text(phase_text), self.styles['CustomBodyText']))
        
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _create_main_content(self, report) -> List:
        elements = []
        
        elements.append(Paragraph("RESEARCH FINDINGS", self.styles['SectionHeader']))
        
        if hasattr(report, 'content') and report.content:
            content = self._sanitize_text(report.content)
            
            paragraphs = content.split('\n\n')
            
            for paragraph in paragraphs:
                if paragraph.strip():
                    if paragraph.strip().startswith('###'):
                        header_text = paragraph.replace('###', '').strip()
                        elements.append(Paragraph(header_text, self.styles['SubHeader']))
                    elif paragraph.strip().startswith('##'):
                        header_text = paragraph.replace('##', '').strip()
                        elements.append(Paragraph(header_text, self.styles['SectionHeader']))
                    else:
                        elements.append(Paragraph(paragraph.strip(), self.styles['CustomBodyText']))
                        elements.append(Spacer(1, 6))
        
        if hasattr(report, 'quality_metrics') and report.quality_metrics:
            elements.append(Paragraph("Report Quality Metrics", self.styles['SubHeader']))
            
            metrics = report.quality_metrics
            metrics_text = f"""
            Report Completeness: {metrics.get('report_completeness', 'Unknown')}<br/>
            Total Sections: {metrics.get('total_sections', '0')}<br/>
            High Confidence Sections: {metrics.get('high_confidence_sections', '0')}<br/>
            Word Count: {metrics.get('word_count', 'Not calculated')}
            """
            
            elements.append(Paragraph(metrics_text, self.styles['CustomBodyText']))
        
        return elements
    
    def _create_citations_section(self, citation_results: Dict[str, Any]) -> List:
        elements = []
        
        elements.append(PageBreak())
        elements.append(Paragraph("REFERENCES AND CITATIONS", self.styles['SectionHeader']))
        
        if citation_results.get('success') and citation_results.get('citations'):
            citations = citation_results['citations']
            
            elements.append(Paragraph(f"Total Citations Found: {len(citations)}", 
                                    self.styles['CustomBodyText']))
            elements.append(Spacer(1, 12))
            
            for i, citation in enumerate(citations[:20], 1):
                if hasattr(citation, 'to_apa'):
                    citation_text = f"{i}. {citation.to_apa()}"
                    if hasattr(citation, 'url') and citation.url:
                        citation_text += f" Available at: {citation.url}"
                else:
                    citation_text = f"{i}. {str(citation)}"
                
                elements.append(Paragraph(self._sanitize_text(citation_text), 
                                        self.styles['Citation']))
                elements.append(Spacer(1, 4))
        else:
            elements.append(Paragraph("No citations were successfully gathered during this research.",
                                    self.styles['CustomBodyText']))
        
        return elements
    
    def _create_appendix(self, results: Dict[str, Any]) -> List:
        elements = []
        
        elements.append(PageBreak())
        elements.append(Paragraph("APPENDIX", self.styles['SectionHeader']))
        
        exec_summary = results.get('execution_summary', {})
        if exec_summary.get('execution_log'):
            elements.append(Paragraph("Execution Log", self.styles['SubHeader']))
            
            log_entries = exec_summary['execution_log'][-10:]
            for entry in log_entries:
                elements.append(Paragraph(self._sanitize_text(entry), self.styles['Citation']))
        
        if exec_summary.get('errors'):
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Errors Encountered", self.styles['SubHeader']))
            
            for error in exec_summary['errors']:
                elements.append(Paragraph(f"• {self._sanitize_text(error)}", 
                                        self.styles['Citation']))
        
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Technical Specifications", self.styles['SubHeader']))
        
        tech_specs = f"""
        Research Framework: LangGraph Multi-Agent System<br/>
        Agents Employed: Planning, Research, Citation, Report Generation<br/>
        Execution Mode: Asynchronous Multi-threaded<br/>
        Quality Validation: Automated Assessment<br/>
        Report Format: PDF with structured sections<br/>
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        elements.append(Paragraph(tech_specs, self.styles['CustomBodyText']))
        
        return elements

class DeepResearchWorkflow:
    def __init__(self):
        self.planner = AgriculturalPlanningAgent()
        self.subsearch_agent = EnhancedSubsearchAgent(max_workers=4)
        self.citation_agent = EnhancedCitationAgent()
        self.report_agent = EnhancedReportAgent()
        self.pdf_generator = PDFReportGenerator()
        
        self.graph = self._build_workflow_graph()
        self.app = self.graph.compile(checkpointer=MemorySaver())

    def _build_workflow_graph(self) -> StateGraph:
        workflow = StateGraph(ResearchState)
        
        workflow.add_node("planning", self._planning_phase)
        workflow.add_node("research_execution", self._research_execution_phase)
        workflow.add_node("citation_gathering", self._citation_gathering_phase)
        workflow.add_node("report_generation", self._report_generation_phase)
        workflow.add_node("quality_validation", self._quality_validation_phase)
        
        workflow.set_entry_point("planning")
        
        workflow.add_edge("planning", "research_execution")
        workflow.add_edge("research_execution", "citation_gathering")
        workflow.add_edge("citation_gathering", "report_generation")
        workflow.add_edge("report_generation", "quality_validation")
        workflow.add_edge("quality_validation", END)
        
        return workflow

    def _log_phase(self, state: ResearchState, phase: str, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {phase.upper()}: {message}"
        state.execution_log.append(log_entry)
        logger.info(log_entry)

    def _planning_phase(self, state: ResearchState) -> ResearchState:
        self._log_phase(state, "planning", "Starting research planning phase")
        state.current_phase = "planning"
        
        try:
            plan = self.planner.create_plan(state.title, state.objective)
            state.plan = plan
            state.tasks = plan.tasks
            
            self._log_phase(state, "planning", f"Generated {len(plan.tasks)} research tasks")
            self._log_phase(state, "planning", f"Agent assignments: {len(plan.agent_assignments)} specialists")
            
        except Exception as e:
            error_msg = f"Planning phase failed: {str(e)}"
            state.errors.append(error_msg)
            self._log_phase(state, "planning", error_msg)
        
        return state

    def _research_execution_phase(self, state: ResearchState) -> ResearchState:
        self._log_phase(state, "research", "Starting research execution phase")
        state.current_phase = "research_execution"
        
        try:
            if not state.tasks:
                self._log_phase(state, "research", "No tasks available, using objective for research")
                research_queries = [f"{state.title} {state.objective}"]
            else:
                research_queries = []
                for task in state.tasks:
                    if hasattr(task, 'subsearch_queries') and task.subsearch_queries:
                        research_queries.extend(task.subsearch_queries[:2])
                    elif hasattr(task, 'description'):
                        research_queries.append(task.description)
                
                if not research_queries:
                    research_queries = [f"{state.title} {state.objective}"]
            
            research_queries = research_queries[:8]
            
            research_results = self.subsearch_agent.search_optimized(research_queries)
            
            state.research_results = research_results
            
            if research_results.get('success'):
                self._log_phase(state, "research", f"Research completed: {research_results.get('successful_searches', 0)} successful queries")
            else:
                self._log_phase(state, "research", "Research execution completed with limited results")
            
        except Exception as e:
            error_msg = f"Research execution failed: {str(e)}"
            state.errors.append(error_msg)
            self._log_phase(state, "research", error_msg)
        
        return state

    def _citation_gathering_phase(self, state: ResearchState) -> ResearchState:
        self._log_phase(state, "citations", "Starting citation gathering phase")
        state.current_phase = "citation_gathering"
        
        try:
            search_topic = f"{state.title} {state.objective}"
            num_citations = 15
            
            citation_results = self.citation_agent.find_citations_basic(
                topic=search_topic,
                num_citations=num_citations
            )
            
            state.citation_results = citation_results
            
            valid_citations = citation_results.get('valid_count', 0) if citation_results.get('success') else 0
            self._log_phase(state, "citations", f"Citation search completed: {valid_citations} valid citations found")
            
        except Exception as e:
            error_msg = f"Citation gathering failed: {str(e)}"
            state.errors.append(error_msg)
            self._log_phase(state, "citations", error_msg)
        
        return state

    def _report_generation_phase(self, state: ResearchState) -> ResearchState:
        self._log_phase(state, "report", "Starting comprehensive report generation")
        state.current_phase = "report_generation"
        
        try:
            research_data = ""
            if state.research_results and state.research_results.get('success'):
                research_data = state.research_results.get('combined_content', '')
            
            citation_data = ""
            if state.citation_results and state.citation_results.get('success'):
                citations = state.citation_results.get('citations', [])
                citation_data = "\n".join([f"- {c.to_apa()} ({c.url})" for c in citations[:10] if hasattr(c, 'to_apa')])
            
            if not research_data and not citation_data:
                research_data = f"Research objective: {state.objective}\nLocation: {state.location}"
                if state.focus_areas:
                    research_data += f"\nFocus areas: {', '.join(state.focus_areas)}"
            
            report = self.report_agent.generate_comprehensive_report(
                title=state.title,
                objective=state.objective,
                research_data=research_data,
                citation_data=citation_data,
                location=state.location,
                focus_areas=state.focus_areas
            )
            
            state.final_report = report
            
            self._log_phase(state, "report", f"Report generated successfully: {report.report_id}")
            if hasattr(report, 'quality_metrics'):
                self._log_phase(state, "report", f"Report sections: {report.quality_metrics.get('total_sections', '0')}")
            
        except Exception as e:
            error_msg = f"Report generation failed: {str(e)}"
            state.errors.append(error_msg)
            self._log_phase(state, "report", error_msg)
        
        return state

    def _quality_validation_phase(self, state: ResearchState) -> ResearchState:
        self._log_phase(state, "validation", "Starting quality validation phase")
        state.current_phase = "quality_validation"
        
        try:
            validation_results = self._validate_research_quality(state)
            
            state.execution_log.append("=== QUALITY VALIDATION RESULTS ===")
            for metric, value in validation_results.items():
                state.execution_log.append(f"{metric}: {value}")
            
            self._log_phase(state, "validation", f"Quality validation completed: {validation_results.get('overall_quality', 'unknown')}")
            
        except Exception as e:
            error_msg = f"Quality validation failed: {str(e)}"
            state.errors.append(error_msg)
            self._log_phase(state, "validation", error_msg)
        
        return state

    def _validate_research_quality(self, state: ResearchState) -> Dict[str, Any]:
        validation = {
            "plan_generated": state.plan is not None,
            "tasks_created": len(state.tasks) if state.tasks else 0,
            "research_successful": state.research_results.get('success', False) if state.research_results else False,
            "citations_found": state.citation_results.get('valid_count', 0) if state.citation_results else 0,
            "report_generated": state.final_report is not None,
            "errors_encountered": len(state.errors),
            "execution_phases_completed": len([log for log in state.execution_log if "Starting" in log])
        }
        
        if state.research_results:
            validation.update({
                "sources_found": state.research_results.get('total_sources_found', 0),
                "successful_searches": state.research_results.get('successful_searches', 0),
                "research_execution_time": state.research_results.get('total_execution_time', 0)
            })
        
        if state.final_report and hasattr(state.final_report, 'quality_metrics'):
            validation.update({
                "report_completeness": state.final_report.quality_metrics.get('report_completeness', 'unknown'),
                "report_sections": int(state.final_report.quality_metrics.get('total_sections', '0')),
                "high_confidence_sections": int(state.final_report.quality_metrics.get('high_confidence_sections', '0'))
            })
        
        quality_score = 0
        if validation["plan_generated"]: quality_score += 20
        if validation["tasks_created"] >= 5: quality_score += 20
        if validation["research_successful"]: quality_score += 25
        if validation["citations_found"] >= 5: quality_score += 20
        if validation["report_generated"]: quality_score += 15
        
        if validation["errors_encountered"] == 0:
            quality_score = min(100, quality_score)
        else:
            quality_score = max(0, quality_score - (validation["errors_encountered"] * 10))
        
        validation["quality_score"] = quality_score
        
        if quality_score >= 90:
            validation["overall_quality"] = "excellent"
        elif quality_score >= 75:
            validation["overall_quality"] = "good"
        elif quality_score >= 60:
            validation["overall_quality"] = "satisfactory"
        else:
            validation["overall_quality"] = "needs_improvement"
        
        return validation

    def _generate_title_from_objective(self, objective: str) -> str:
        objective_lower = objective.lower()
        
        if any(word in objective_lower for word in ['improve', 'increase', 'enhance', 'optimize']):
            if any(word in objective_lower for word in ['yield', 'production', 'harvest']):
                return f"Agricultural Yield Enhancement Study: {objective}"
            elif any(word in objective_lower for word in ['soil', 'fertility', 'nutrient']):
                return f"Soil Management Optimization Research: {objective}"
            else:
                return f"Agricultural Improvement Analysis: {objective}"
        
        elif any(word in objective_lower for word in ['evaluate', 'assess', 'analyze', 'study']):
            if any(word in objective_lower for word in ['technology', 'innovation', 'system']):
                return f"Agricultural Technology Assessment: {objective}"
            elif any(word in objective_lower for word in ['sustainability', 'environment', 'climate']):
                return f"Sustainable Agriculture Research: {objective}"
            else:
                return f"Agricultural Research Study: {objective}"
        
        elif any(word in objective_lower for word in ['develop', 'design', 'create']):
            return f"Agricultural Development Research: {objective}"
        
        elif any(word in objective_lower for word in ['compare', 'contrast']):
            return f"Comparative Agricultural Study: {objective}"
        
        else:
            return f"Agricultural Research: {objective}"

    def execute_research(self, objective: str, location: str = "Global", 
                        focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        title = self._generate_title_from_objective(objective)
        logger.info(f"Starting deep research workflow: {title}")
        
        initial_state = ResearchState(
            title=title,
            objective=objective,
            location=location,
            focus_areas=focus_areas or []
        )
        
        try:
            config = {"configurable": {"thread_id": f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
            
            final_state = initial_state
            
            for state_update in self.app.stream(initial_state, config):
                if isinstance(state_update, dict):
                    for key, state_obj in state_update.items():
                        if hasattr(state_obj, "title") and hasattr(state_obj, "objective"):
                            final_state = state_obj
                            break
                elif hasattr(state_update, "title") and hasattr(state_update, "objective"):
                    final_state = state_update
                
                current_phase = getattr(final_state, 'current_phase', 'unknown')
                logger.info(f"Completed phase: {current_phase}")
                    
            return self._format_final_results(final_state)
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "title": title,
                "objective": objective,
                "location": location,
                "focus_areas": focus_areas or [],
                "planning_results": {},
                "research_results": {},
                "citation_results": {},
                "final_report": None,
                "execution_summary": {},
                "quality_validation": None
            }

    def _format_final_results(self, state: ResearchState) -> Dict[str, Any]:
        return {
            "success": state.final_report is not None,
            "title": state.title,
            "objective": state.objective,
            "location": state.location,
            "focus_areas": state.focus_areas,
            
            "planning_results": {
                "plan_generated": state.plan is not None,
                "tasks_created": len(state.tasks) if state.tasks else 0,
                "agent_assignments": len(state.plan.agent_assignments) if state.plan else 0
            },
            
            "research_results": state.research_results,
            "citation_results": state.citation_results,
            "final_report": state.final_report,
            
            "execution_summary": {
                "phases_completed": len([log for log in state.execution_log if "Starting" in log]),
                "errors_encountered": len(state.errors),
                "execution_log": state.execution_log,
                "errors": state.errors
            },
            
            "quality_validation": self._validate_research_quality(state) if state.current_phase == "quality_validation" else None
        }

    def save_pdf_report(self, results: Dict[str, Any], custom_filename: str = None) -> str:
        try:
            if custom_filename:
                filename = custom_filename
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
            else:
                safe_objective = results.get('objective', 'research').replace(' ', '_')[:30]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                filename = f"Reports/agricultural_research_{safe_objective}_{timestamp}.pdf"
            
            if not filename.startswith('Reports/'):
                filename = f"Reports/{filename}"
            
            self.pdf_generator.generate_pdf_report(results, filename)
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save PDF report: {str(e)}")
            raise e

    def display_results(self, results: Dict[str, Any]):
        print(f"\nDEEP RESEARCH WORKFLOW RESULTS")
        print("=" * 80)
        print(f"Title: {results['title']}")
        print(f"Objective: {results['objective']}")
        print(f"Location: {results['location']}")
        if results.get('focus_areas'):
            print(f"Focus Areas: {', '.join(results['focus_areas'])}")
        
        print(f"\nWorkflow Success: {'Yes' if results['success'] else 'No'}")
        
        exec_summary = results.get('execution_summary', {})
        print(f"\nEXECUTION SUMMARY")
        print(f"Phases Completed: {exec_summary.get('phases_completed', 0)}/5")
        print(f"Errors Encountered: {exec_summary.get('errors_encountered', 0)}")
        
        planning = results.get('planning_results', {})
        print(f"\nPLANNING PHASE")
        print(f"Plan Generated: {'Yes' if planning.get('plan_generated') else 'No'}")
        print(f"Tasks Created: {planning.get('tasks_created', 0)}")
        print(f"Agent Assignments: {planning.get('agent_assignments', 0)}")
        
        research = results.get('research_results', {})
        if research:
            print(f"\nRESEARCH PHASE")
            print(f"Research Success: {'Yes' if research.get('success') else 'No'}")
            print(f"Successful Searches: {research.get('successful_searches', 0)}/{research.get('queries_processed', 0)}")
            print(f"Sources Found: {research.get('total_sources_found', 0)}")
            print(f"Execution Time: {research.get('total_execution_time', 0):.2f}s")
        
        citations = results.get('citation_results', {})
        if citations:
            print(f"\nCITATION PHASE")
            print(f"Citation Success: {'Yes' if citations.get('success') else 'No'}")
            print(f"Valid Citations: {citations.get('valid_count', 0)}")
            print(f"Total Sources Examined: {citations.get('total_found', 0)}")
        
        if results.get('final_report'):
            report = results['final_report']
            print(f"\nREPORT GENERATION")
            print(f"Report Generated: Yes")
            print(f"Report ID: {report.report_id}")
            
            if hasattr(report, 'quality_metrics'):
                metrics = report.quality_metrics
                print(f"Report Quality: {metrics.get('report_completeness', 'unknown').upper()}")
                print(f"Sections: {metrics.get('total_sections', '0')}")
                print(f"High Confidence: {metrics.get('high_confidence_sections', '0')}")
        
        quality = results.get('quality_validation', {})
        if quality:
            print(f"\nQUALITY VALIDATION")
            print(f"Overall Quality: {quality.get('overall_quality', 'unknown').upper()}")
            print(f"Quality Score: {quality.get('quality_score', 0)}/100")
        
        if exec_summary.get('errors'):
            print(f"\nERRORS ENCOUNTERED")
            for error in exec_summary['errors']:
                print(f"- {error}")
        
        if results.get('final_report'):
            print(f"\nREPORT PREVIEW")
            print("-" * 80)
            formatted_report = self.report_agent.format_report_display(results['final_report'])
            preview = formatted_report[:1000]
            if len(formatted_report) > 1000:
                preview += "\n\n... [Full report available in PDF] ..."
            print(preview)


def main():
    print("Deep Agricultural Research Pipeline")
    print("=" * 50)
    
    try:
        import reportlab
        print("PDF generation support available")
    except ImportError:
        print("ReportLab not found. Installing...")
        print("Run: pip install reportlab")
        sys.exit(1)
    
    workflow = DeepResearchWorkflow()
    
    objective = input("Research Objective: ").strip()
    if not objective:
        objective = "rice cultivation scope in North east India"
        print(f"Using default objective: {objective}")
    
    location = input("Location (optional): ").strip() or "Global"
    
    focus_input = input("Focus Areas (comma-separated, optional): ").strip()
    focus_areas = [area.strip() for area in focus_input.split(",")] if focus_input else []
    
    generated_title = workflow._generate_title_from_objective(objective)
    print(f"\nGenerated Title: {generated_title}")
    
    print(f"\nStarting deep research workflow...")
    print(f"This will involve: Planning -> Multi-threaded Research -> Citation Gathering -> Report Generation -> Quality Validation")
    print("Please wait, this may take several minutes...\n")
    
    results = workflow.execute_research(
        objective=objective,
        location=location,
        focus_areas=focus_areas
    )
    
    workflow.display_results(results)
    
    save_option = input(f"\nSave comprehensive PDF report? (y/n): ").strip().lower()
    if save_option == 'y':
        try:
            custom_name = input("Custom filename (optional, press Enter for auto-generated): ").strip()
            
            if custom_name:
                filename = workflow.save_pdf_report(results, custom_name)
            else:
                filename = workflow.save_pdf_report(results)
            
            print(f"PDF report saved successfully!")
            print(f"Location: {filename}")
            print(f"Report includes: Title page, Executive summary, Research findings, Citations, and Technical appendix")
            
        except Exception as e:
            print(f"Failed to save PDF: {str(e)}")
            print("Make sure ReportLab is installed: pip install reportlab")
    
    json_backup = input(f"\nAlso save JSON backup for data analysis? (y/n): ").strip().lower()
    if json_backup == 'y':
        try:
            import json
            safe_objective = objective.replace(' ', '_')[:30]
            json_filename = f"Reports/research_data_{safe_objective}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"JSON backup saved to: {json_filename}")
        except Exception as e:
            print(f"Failed to save JSON backup: {str(e)}")

if __name__ == "__main__":
    main()