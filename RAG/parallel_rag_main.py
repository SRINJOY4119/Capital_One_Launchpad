import os
import sys
import asyncio
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any
import time
import hashlib

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

cache_base_dir = os.path.join(current_dir, "parallel_cache")
os.makedirs(cache_base_dir, exist_ok=True)

from dotenv import load_dotenv
from workflow import Workflow
from agno.agent import Agent
from agno.models.google import Gemini

load_dotenv()

class ParallelRAGSystem:
    def __init__(self, model="llama-3.3-70b-versatile", k=3):
        self.model = model
        self.k = k
        self.api_key = os.getenv("GROQ_API_KEY")
        self.data_dir = Path(current_dir) / "Data"
        self.cache_base = Path(cache_base_dir)
        
        self.synthesizer = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            show_tool_calls=False,
            markdown=True,
            instructions="""You are an expert information synthesizer specializing in agricultural and research data analysis. Your role is to combine and synthesize information from multiple RAG workflow outputs to provide the most comprehensive and accurate response.

**SYNTHESIS GUIDELINES:**

1. **Analyze Multiple Sources**: Review all RAG outputs from different documents/datasets
2. **Identify Common Themes**: Find overlapping information and consistent patterns
3. **Resolve Contradictions**: When sources disagree, prioritize based on:
   - Source credibility and recency
   - Specificity and detail level
   - Supporting evidence quality
4. **Combine Complementary Information**: Merge unique insights from different sources
5. **Maintain Source Attribution**: Reference which sources provided specific information

**OUTPUT STRUCTURE:**
- **Summary**: Concise overview of the synthesized answer
- **Key Findings**: Main points from multiple sources
- **Source Breakdown**: What each source contributed
- **Confidence Assessment**: How reliable the synthesized answer is
- **Recommendations**: Actionable insights based on combined information

**QUALITY STANDARDS:**
- Prioritize accuracy over completeness
- Clearly distinguish between confirmed facts and assumptions
- Highlight any gaps or uncertainties in the information
- Provide balanced perspectives when sources offer different viewpoints"""
        )
    
    def get_data_files(self) -> List[Path]:
        supported_extensions = ['.csv', '.pdf']
        data_files = []
        
        if not self.data_dir.exists():
            print(f"Data directory not found: {self.data_dir}")
            return data_files
        
        search_locations = [
            self.data_dir / "CSV",
        ]
        
        print(f"Searching for files in:")
        for location in search_locations:
            exists = "✓" if location.exists() else "✗"
            if location.exists():
                pdf_count = len(list(location.glob("*.pdf")))
                csv_count = len(list(location.glob("*.csv")))
                print(f"   {exists} {location} ({pdf_count} PDFs, {csv_count} CSVs)")
            else:
                print(f"   {exists} {location}")
        
        for location in search_locations:
            if location.exists():
                for ext in supported_extensions:
                    for file_path in location.glob(f"*{ext}"):
                        if file_path.is_file() and file_path not in data_files:
                            # Include chunk files as they are processed documents
                            data_files.append(file_path)
        
        return data_files
    
    def get_file_cache_id(self, file_path: Path) -> str:
        file_info = f"{file_path.name}_{file_path.stat().st_size}_{file_path.stat().st_mtime}"
        return hashlib.md5(file_info.encode()).hexdigest()[:8]
    
    def setup_workflow_cache(self, file_path: Path) -> str:
        cache_id = self.get_file_cache_id(file_path)
        workflow_cache_dir = self.cache_base / f"workflow_{cache_id}"
        workflow_cache_dir.mkdir(exist_ok=True)
        return str(workflow_cache_dir)
    
    def run_single_workflow(self, file_path: Path, question: str) -> Dict[str, Any]:
        try:
            print(f"Processing: {file_path.name}")
            
            workflow_cache_dir = self.setup_workflow_cache(file_path)
            
            workflow = Workflow(
                model=self.model, 
                api_key=self.api_key, 
                k=self.k, 
                file_path=str(file_path),
                cache_dir=workflow_cache_dir  
            )
            
            inputs = {"question": question}
            
            start_time = time.time()
            result = workflow.run_workflow(inputs)
            end_time = time.time()
            
            return {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix,
                "success": True,
                "response": result.get('generation', ''),
                "processing_time": end_time - start_time,
                "workflow_type": result.get('workflow_type', 'standard'),
                "extractions": result.get('extractions', ''),
                "cache_dir": workflow_cache_dir,
                "error": None
            }
            
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {str(e)}")
            return {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix,
                "success": False,
                "response": "",
                "processing_time": 0,
                "workflow_type": "failed",
                "extractions": "",
                "cache_dir": "",
                "error": str(e)
            }
    
    def run_parallel_workflows(self, question: str, max_workers: int = None) -> List[Dict[str, Any]]:
        data_files = self.get_data_files()
        
        if not data_files:
            print("No supported data files found (CSV or PDF)")
            return []
        
        if max_workers is None or max_workers > len(data_files):
            max_workers = len(data_files)
        
        print(f"Found {len(data_files)} files to process:")
        for file_path in data_files:
            print(f"   • {file_path.name} ({file_path.suffix})")
        
        print(f"\nStarting parallel processing with {max_workers} workers...")
        print("⚡ Early stopping enabled: Will proceed to synthesis after 5 successful results")
        
        results = []
        successful_count = 0
        early_stop_threshold = 400
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.run_single_workflow, file_path, question): file_path
                for file_path in data_files
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["success"]:
                        successful_count += 1
                    
                    status = "✅" if result["success"] else "❌"
                    print(f"{status} Completed: {file_path.name} ({successful_count}/{early_stop_threshold} successful)")
                    
                    # Early stopping condition
                    if successful_count >= early_stop_threshold:
                        print(f"🎯 Early stopping triggered! Got {successful_count} successful results.")
                        print("🔄 Cancelling remaining tasks and proceeding to synthesis...")
                        
                        # Cancel remaining futures
                        for remaining_future in future_to_file:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        
                        break
                        
                except Exception as e:
                    print(f"❌ Exception for {file_path.name}: {str(e)}")
                    results.append({
                        "file_name": file_path.name,
                        "success": False,
                        "error": str(e)
                    })
        
        print(f"✅ Processing completed with {successful_count} successful results out of {len(results)} processed files")
        return results
    
    def synthesize_results(self, question: str, workflow_results: List[Dict[str, Any]]) -> str:
        successful_results = [r for r in workflow_results if r["success"] and r["response"]]
        
        if not successful_results:
            return "No successful workflow results to synthesize."
        
        synthesis_prompt = f"""
**ORIGINAL QUESTION:** {question}

**RAG WORKFLOW RESULTS TO SYNTHESIZE:**

"""
        
        for i, result in enumerate(successful_results, 1):
            synthesis_prompt += f"""
**Source {i}: {result['file_name']} ({result['file_type']})**
- Processing Time: {result['processing_time']:.2f}s
- Workflow Type: {result['workflow_type']}
- Response: {result['response']}
"""
            if result['extractions']:
                synthesis_prompt += f"- Key Extractions: {result['extractions']}\n"
            
            synthesis_prompt += "---\n"
        
        synthesis_prompt += f"""

**SYNTHESIS TASK:**
Analyze all the above responses and provide a comprehensive, synthesized answer to the original question: "{question}"

Consider information quality, source reliability, and complementary insights. Resolve any contradictions and highlight the most valuable information from across all sources.
"""
        
        print("Synthesizing results from multiple sources...")
        synthesized_response = self.synthesizer.run(synthesis_prompt)
        
        return synthesized_response.content
    
    def process_query(self, question: str, max_workers: int = None) -> Dict[str, Any]:
        print("=" * 80)
        print("🌾 PARALLEL RAG SYSTEM")
        print("=" * 80)
        
        start_time = time.time()
        
        workflow_results = self.run_parallel_workflows(question, max_workers)
        
        synthesized_answer = self.synthesize_results(question, workflow_results)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        successful_count = sum(1 for r in workflow_results if r["success"])
        failed_count = len(workflow_results) - successful_count
        
        return {
            "question": question,
            "total_files_processed": len(workflow_results),
            "successful_workflows": successful_count,
            "failed_workflows": failed_count,
            "individual_results": workflow_results,
            "synthesized_answer": synthesized_answer,
            "total_processing_time": total_time,
            "average_time_per_file": total_time / len(workflow_results) if workflow_results else 0
        }
    
    def clear_all_caches(self):
        """Clear all workflow caches."""
        import shutil
        if self.cache_base.exists():
            shutil.rmtree(self.cache_base)
            print(f"🗑️ Cleared all parallel workflow caches")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cache usage."""
        if not self.cache_base.exists():
            return {"total_caches": 0, "total_size_mb": 0}
        
        cache_dirs = list(self.cache_base.glob("workflow_*"))
        total_size = 0
        
        for cache_dir in cache_dirs:
            for file_path in cache_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        
        return {
            "total_caches": len(cache_dirs),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_directories": [d.name for d in cache_dirs]
        }

def main():
    parallel_rag = ParallelRAGSystem(
        model="llama-3.3-70b-versatile", 
        k=3
    )
    
    cache_info = parallel_rag.get_cache_info()
    print(f"📁 Cache Info: {cache_info['total_caches']} caches, {cache_info['total_size_mb']:.2f} MB")
    
    test_questions = [
        "What are the main agricultural practices mentioned in the documents?",
        "Provide information about crop recommendations and soil requirements",
        "What research findings or methodologies are discussed?",
        "Summarize the key insights about farming techniques and technologies"
    ]
    
    question = test_questions[0]
    print(f"Processing Question: {question}")
    
    result = parallel_rag.process_query(question)
    
    cache_info_after = parallel_rag.get_cache_info()
    print(f"\n📁 Cache Info After: {cache_info_after['total_caches']} caches, {cache_info_after['total_size_mb']:.2f} MB")
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Question: {result['question']}")
    print(f"Files Processed: {result['total_files_processed']}")
    print(f"Successful: {result['successful_workflows']}")
    print(f"Failed: {result['failed_workflows']}")
    print(f"Total Time: {result['total_processing_time']:.2f}s")
    print(f"Avg Time/File: {result['average_time_per_file']:.2f}s")
    
    print("\nSYNTHESIZED ANSWER:")
    print("-" * 60)
    print(result['synthesized_answer'])
    
    print("\nINDIVIDUAL WORKFLOW RESULTS:")
    print("-" * 60)
    for res in result['individual_results']:
        status = "✅" if res['success'] else "❌"
        print(f"{status} {res['file_name']}: {res['processing_time']:.2f}s")
        if not res['success']:
            print(f"   Error: {res['error']}")

if __name__ == "__main__":
    main()
