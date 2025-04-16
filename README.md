# Deep Research Model Comparison

This tool is a Streamlit-based web application designed to compare the research output of different language models (Gemini, OpenAI, and Ollama) in a deep research context.

## Features

- **Concurrent Research**: Run research queries concurrently across multiple models.
- **API Key Configuration**: Requires a mandatory Serper API key and supports optional OpenAI and Gemini API keys.
- **Extensible Agents**: Uses specialized agents for deep research, analysis, and report writing.
- **Results Comparison**: Displays a detailed comparison of model responses and saves results as JSON.

## Installation

1. **Prerequisites**:  
   - Python 3.10 or higher  
   - Pip package manager

2. **Dependencies**:  
   Install the required packages by running:
   ```bash
   pip install -r New_Research_Agent_By_DifferentLLMS/requirements.txt
   ```

## Usage

1. **Start the Application**:  
   Run the Streamlit app from the terminal:
   ```bash
   streamlit run New_Research_Agent/NewComparisonbydifferentLLMS.py
   ```
   
2. **Configure API Keys**:  
   - Enter your **Serper API Key** (mandatory) in the sidebar.
   - Optionally provide your **OpenAI** and **Gemini API Keys**.

3. **Run Research**:  
   - Input your research query.
   - Select the models you wish to compare.
   - Click **Start Research** to initiate the process.

4. **View & Save Results**:  
   - Results are displayed in separate columns with progress indicators.
   - Final outputs are automatically saved as JSON files in the project directory.

## Project Structure

- `New_Research_Agent_By_DifferentLLMS/NewComparisonbydifferentLLMS.py`: Main application file.
- `New_Research_Agent_By_DifferentLLMS/requirements.txt`: Dependency list.
- `research_results_<timestamp>.json`: Auto-generated file storing research outcomes.


## Contact

For questions or support, please open an issue in the repository or contact the project maintainers.