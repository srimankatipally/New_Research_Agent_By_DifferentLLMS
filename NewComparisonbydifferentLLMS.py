import os
# Removed dotenv loading and environment variable settings
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, WebsiteSearchTool
import streamlit as st
import concurrent.futures
from datetime import datetime
import pandas as pd
import json

def get_llm(model_choice='gemini', api_keys=None):
    """Get the specified language model using provided API keys"""
    if api_keys is None:
        api_keys = {}
    if model_choice == 'openai':
        return LLM(
            model="openai/o1-mini",
            api_key=api_keys.get("openai_api_key"),
            verbose=True
        )
    elif model_choice == 'gemini':
        return LLM(
            model="gemini/gemini-2.0-flash",
            temperature=0.7,
            google_api_key=api_keys.get("gemini_api_key"),
            verbose=True
        )
    else:  # ollama
        return LLM(
            model="ollama/deepseek-r1:latest",
            base_url="http://localhost:11434",
        )

def create_agents(model_choice, api_keys):
    """Create specialized research and analysis agents"""
    llm = get_llm(model_choice, api_keys)
    
    deep_researcher = Agent(
        role='Deep Research Specialist',
        goal='Conduct comprehensive internet research and data gathering',
        backstory="""Expert at conducting deep, thorough research across multiple sources. 
        Skilled at finding hard-to-locate information and connecting disparate data points. 
        Specializes in complex research tasks that would typically take hours or days.""",
        tools=[search_tool, website_tool],
        llm=llm,
        verbose=True,
        max_iter=100,
        allow_delegation=False,
        max_rpm=50,
        max_retry_limit=3
    )
    
    analyst = Agent(
        role='Research Analyst',
        goal='Analyze and synthesize complex research findings',
        backstory="""Expert analyst skilled at processing large amounts of information,
        identifying patterns, and drawing meaningful conclusions. Specializes in turning
        raw research into actionable insights.""",
        tools=[search_tool],
        llm=llm,
        verbose=True,
        max_iter=75,
        allow_delegation=False,
        max_rpm=30,
        max_retry_limit=2
    )
    
    report_writer = Agent(
        role='Research Report Writer',
        goal='Create comprehensive, well-structured research reports',
        backstory="""Expert at transforming complex research and analysis into 
        clear, actionable reports. Skilled at maintaining detail while ensuring 
        accessibility and practical value.""",
        llm=llm,
        verbose=True,
        max_iter=50,
        allow_delegation=False,
        max_rpm=20,
        max_retry_limit=2
    )
    
    return deep_researcher, analyst, report_writer

def create_tasks(researcher, analyst, writer, research_query):
    """Create research tasks with clear objectives"""
    deep_research_task = Task(
        description=f"""Conduct focused research on: {research_query}
        
        Step-by-step approach:
        1. Initial broad search to identify key sources
        2. Deep dive into most relevant sources
        3. Extract specific details and evidence
        4. Verify key findings across sources
        5. Document sources and findings clearly
        
        Keep focused on specific, verified information.""",
        agent=researcher,
        expected_output="Detailed research findings with verified sources"
    )
    
    analysis_task = Task(
        description=f"""Analyze the research findings about {research_query}:
        
        Follow these steps:
        1. Review and categorize all findings
        2. Identify main themes and patterns
        3. Evaluate source credibility
        4. Note any inconsistencies
        5. Summarize key insights
        
        Focus on clear, actionable analysis.""",
        agent=analyst,
        context=[deep_research_task],
        expected_output="Clear analysis of findings with key insights"
    )
    
    report_task = Task(
        description=f"""Create a structured report about {research_query}:
        
        Include:
        1. Executive summary (2-3 paragraphs)
        2. Key findings (bullet points)
        3. Supporting evidence
        4. Conclusions
        5. References
        
        Keep it clear and focused.""",
        agent=writer,
        context=[deep_research_task, analysis_task],
        expected_output="Concise, well-structured report"
    )
    
    return [deep_research_task, analysis_task, report_task]

def create_crew(agents, tasks):
    """Create a crew with optimal settings"""
    return Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,
        max_rpm=100,
        process="sequential"
    )

def run_research(model_choice, query, api_keys):
    """Run research with specified model and return results"""
    try:
        start_time = datetime.now()
        researcher, analyst, writer = create_agents(model_choice, api_keys)
        tasks = create_tasks(researcher, analyst, writer, query)
        crew = create_crew([researcher, analyst, writer], tasks)
        result = crew.kickoff()
        execution_time = (datetime.now() - start_time).total_seconds()
        return {'result': result, 'execution_time': execution_time}
    except Exception as e:
        return f"Error with {model_choice}: {str(e)}"

class CrewOutputEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "text") and hasattr(obj, "metadata"):
            return {
                "text": obj.text,
                "metadata": obj.metadata
            }
        return super().default(obj)

def save_results(query, results):
    """Save research results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_results_{timestamp}.json"
    
    data = {
        "query": query,
        "timestamp": timestamp,
        "results": results
    }
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, default=str)
    
    return filename

def main():
    st.set_page_config(page_title="Research Model Comparison", layout="wide")
    
    st.title("🔍 Deep Research Model Comparison")
    
    # Sidebar configuration for API keys
    st.sidebar.header("API Configuration")
    serper_api_key = st.sidebar.text_input("Serper API Key (Mandatory)", type="password")
    openai_api_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password")
    gemini_api_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password")
    st.sidebar.header("Connect With Me")
    st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-srimankatipally-black?style=flat&logo=github)](https://github.com/srimankatipally) [![LinkedIn](https://img.shields.io/badge/LinkedIn-srimankatipally-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/srimankatipally/)")

    if not serper_api_key:
        st.error("Serper API Key is required!")
        st.stop()
    
    # Build API keys dictionary
    api_keys = {
        "serper_api_key": serper_api_key,
        "openai_api_key": openai_api_key,
        "gemini_api_key": gemini_api_key
    }
    
    # Initialize enhanced search tools with provided serper key
    global search_tool, website_tool  # Used in create_agents
    search_tool = SerperDevTool(api_key=serper_api_key)
    website_tool = WebsiteSearchTool()
    
    # Sidebar model selection
    st.sidebar.header("Configuration")
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        ["Gemini", "OpenAI", "Ollama"],
        default=["Gemini"]
    )
    
    # Convert display names to internal names
    model_mapping = {
        "Gemini": "gemini",
        "OpenAI": "openai",
        "Ollama": "ollama"
    }
    
    # Main query input
    query = st.text_area("Research Query", height=100, placeholder="Enter your research query here...")
    
    if st.button("Start Research", type="primary"):
        if not query:
            st.error("Please enter a research query")
            return
        
        if not selected_models:
            st.error("Please select at least one model")
            return
        
        # Create progress containers
        progress_bars = {model: st.progress(0) for model in selected_models}
        status_containers = {model: st.empty() for model in selected_models}
        timer_containers = {model: st.empty() for model in selected_models}
        
        # Initialize results dictionary
        results = {}
        
        # Create columns for results
        cols = st.columns(len(selected_models))
        result_containers = {model: cols[i].container() for i, model in enumerate(selected_models)}
        
        start_times = {model: None for model in selected_models}
        
        # Run research for each selected model
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_model = {
                executor.submit(run_research, model_mapping[model], query, api_keys): model 
                for model in selected_models
            }
            
            # Start times for each model
            for model in selected_models:
                start_times[model] = datetime.now()
            
            while future_to_model:
                done, _ = concurrent.futures.wait(future_to_model.keys(), timeout=0.1)
                
                # Update running timers
                for model in selected_models:
                    if model in results:  # Skip if already completed
                        continue
                    current_time = (datetime.now() - start_times[model]).total_seconds()
                    timer_containers[model].text(f"⏱️ Running Time: {current_time:.1f}s")
                
                for future in done:
                    model = future_to_model[future]
                    try:
                        result_data = future.result()
                        if isinstance(result_data, dict):
                            results[model] = result_data['result']
                            execution_time = result_data['execution_time']
                            
                            progress_bars[model].progress(100)
                            status_containers[model].success(f"{model} Research Complete")
                            timer_containers[model].text(f"⏱️ Final Time: {execution_time:.2f}s")
                            
                            with result_containers[model]:
                                st.subheader(f"{model} Results")
                                st.write(results[model])
                        else:
                            progress_bars[model].progress(100)
                            status_containers[model].error(f"Error with {model}: {result_data}")
                            timer_containers[model].empty()
                    except Exception as e:
                        progress_bars[model].progress(100)
                        status_containers[model].error(f"Error with {model}: {str(e)}")
                        timer_containers[model].empty()
                    
                    del future_to_model[future]

        # Save results if any were generated
        if results:
            filename = save_results(query, results)
            st.sidebar.success(f"Results saved to {filename}")
            
            # Create comparison table
            st.subheader("Quick Comparison")
            comparison_data = {
                "Model": list(results.keys()),
                "Response Length": [len(str(r)) for r in results.values()],
                "Contains References": ["References" in str(r) for r in results.values()],
                "Contains Analysis": ["Analysis" in str(r) for r in results.values()]
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)

if __name__ == "__main__":
    main()