import streamlit as st
import pandas as pd
import io
import json
import time
import os
import requests

# --- 1. CONFIGURATION AND UTILITIES ---

# API Key is now sourced ONLY from st.secrets['GEMINI_API_KEY']
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"

# Hardcoded mapping for IMD subdivisions to State Names (Crucial integration logic for Phase 1)
SUBDIVISION_TO_STATE_MAP = {
    'ANDAMAN & NICOBAR ISLAND': 'Andaman & Nicobar Islands',
    'ARUNACHAL PRADESH': 'Arunachal Pradesh',
    'ASSAM & MEGHALAYA': 'Assam',
    'NAGA MANI MIZO TRIPURA': 'Nagaland, Mizoram, Tripura, Manipur',
    'SUB HIMALAYAN WEST BENGAL & SIKKIM': 'West Bengal',
    'GANGETIC WEST BENGAL': 'West Bengal',
    'ORISSA': 'Odisha',
    'BIHAR': 'Bihar',
    'UTTAR PRADESH EAST': 'Uttar Pradesh',
    'UTTAR PRADESH WEST': 'Uttar Pradesh',
    'UTTARAKHAND': 'Uttarakhand',
    'HARYANA DELHI & CHANDIGARH': 'Haryana',
    'PUNJAB': 'Punjab',
    'HIMACHAL PRADESH': 'Himachal Pradesh',
    'JAMMU & KASHMIR': 'Jammu & Kashmir',
    'WEST RAJASTHAN': 'Rajasthan',
    'EAST RAJASTHAN': 'Rajasthan',
    'WEST MADHYA PRADESH': 'Madhya Pradesh',
    'EAST MADHYA PRADESH': 'Madhya Pradesh',
    'GUJARAT REGION': 'Gujarat',
    'SAURASHTRA & KUTCH': 'Gujarat',
    'KONKAN & GOA': 'Maharashtra',
    'MADHYA MAHARASHTRA': 'Maharashtra',
    'MARATHWADA': 'Maharashtra',
    'VIDARBHA': 'Maharashtra',
    'CHHATTISGARH': 'Chhattisgarh',
    'ANDHRA PRADESH': 'Andhra Pradesh',
    'TAMIL NADU': 'Tamil Nadu',
    'COASTAL KARNATAKA': 'Karnataka',
    'NORTH INTERIOR KARNATAKA': 'Karnataka',
    'SOUTH INTERIOR KARNATAKA': 'Karnataka',
    'KERALA': 'Kerala',
    'LAKSHDWEEP': 'Lakshadweep',
    'TELANGANA': 'Telangana',
    'RAYALASEEMA': 'Andhra Pradesh',
    'COASTAL ANDHRA PRADESH': 'Andhra Pradesh',
    'JHARKHAND': 'Jharkhand',
}

# --- 2. DATA PREPROCESSING AND INTEGRATION LOGIC (Phase 1 Solution) ---

@st.cache_data(show_spinner="Integrating and normalizing datasets...")
def preprocess_data(folder_path, rf_filename, lac_filename):
    """
    Loads, cleans, and merges the two disparate datasets from a specified folder path.
    """
    rf_path = os.path.join(folder_path, rf_filename)
    lac_path = os.path.join(folder_path, lac_filename)

    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"Rainfall file not found at: {rf_path}")
    if not os.path.exists(lac_path):
        raise FileNotFoundError(f"Lac Production file not found at: {lac_path}")

    # Load Rainfall Data (IMD)
    rf_df = pd.read_csv(rf_path)
    rf_df.columns = [col.strip() for col in rf_df.columns]
    rf_df = rf_df.rename(columns={'subdivision': 'Subdivision', 'YEAR': 'Year', 'JUN-SEP': 'Rainfall_Monsoon_mm'})
    
    # Geographic Normalization
    rf_df['State'] = rf_df['Subdivision'].str.upper().apply(lambda x: SUBDIVISION_TO_STATE_MAP.get(x.strip(), ''))
    rf_df_monsoon = rf_df[['State', 'Year', 'Rainfall_Monsoon_mm']].copy()
    rf_df_monsoon = rf_df_monsoon[rf_df_monsoon['State'] != '']

    # Load Lac Production Data (Agriculture)
    lac_df = pd.read_csv(lac_path)
    lac_df = lac_df.rename(columns={'States': 'State'})

    # Melt the production data for easier analysis
    prod_melt = lac_df.melt(id_vars=['State'], var_name='Financial_Year', value_name='Lac_Production_Tons')

    # Temporal Normalization
    prod_melt['Year'] = prod_melt['Financial_Year'].apply(lambda x: int(x.split('-')[0]) if pd.notna(x) and '-' in x else None)
    
    # Data Cleaning and Type Conversion
    prod_melt['Lac_Production_Tons'] = pd.to_numeric(
        prod_melt['Lac_Production_Tons'].replace({'NA': None}), 
        errors='coerce'
    )
    prod_df = prod_melt[['State', 'Year', 'Lac_Production_Tons']].dropna()

    # Final Merge: Join the two datasets on the Normalized Keys ('State', 'Year')
    df_unified = pd.merge(
        prod_df,
        rf_df_monsoon,
        on=['State', 'Year'],
        how='inner'
    )
    df_unified = df_unified[df_unified['State'] != 'Total'].reset_index(drop=True)

    # CRITICAL: Filter unified data to only cover the intersecting time period of the two datasets
    # Lac data is 2007-2011 (financial years 2007-08 to 2011-12)
    min_year = df_unified['Year'].min()
    max_year = df_unified['Year'].max()
    
    st.session_state['data_range'] = f"{min_year}-{max_year}"
    
    return df_unified

# --- 3. LLM AGENT LOGIC (Phase 2 Solution) ---

def answer_question(question, df_unified):
    """
    Uses the LLM to generate Python code, executes it, and interprets the result.
    Securely fetches the API key from st.secrets.
    """
    
    # Securely fetch API Key from st.secrets
    try:
        api_key_to_use = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        # This fallback is for local testing if the key is not in secrets.toml
        st.error("Deployment Error: GEMINI_API_KEY not found in secrets.toml.")
        return "System configuration error: Please ensure 'GEMINI_API_KEY' is set in your Streamlit secrets file."


    # 1. System Instruction & Context (The Agent's Persona)
    system_instruction = f"""
    You are Project Samarth, an intelligent cross-domain data analysis and Q&A system for policy advisors.
    Your role is to analyze a unified, cleaned dataset on Indian Agriculture and Climate.
    The data is available in a Pandas DataFrame called 'df_unified'.
    The current available time range is {st.session_state.get('data_range', '2007-2011')}.
    The DataFrame contains the following critical columns:
    - 'State': The administrative state (Normalized Geographic Unit).
    - 'Year': The calendar year (Normalized Temporal Unit, e.g., 2008).
    - 'Rainfall_Monsoon_mm': Total Jun-Sep rainfall in mm (from IMD data).
    - 'Lac_Production_Tons': Production of Lac in Tons (from Agricultural data).

    When the user asks a question, your process is:
    1. **Tool Use (Code Generation):** If the question requires data analysis (comparison, aggregation, statistical analysis, or filtering), you MUST generate a complete Python code snippet inside a ```python block.
    2. **Code Constraints (CRITICAL FIX):** The code MUST ONLY use the 'df_unified' DataFrame. Use standard Pandas operations. **DO NOT use .to_markdown() as this requires external libraries.** Print the final calculated result, a concise summary DataFrame, or a clear textual finding using **.to_string()** or **plain print()** statements to stdout.
    3. **Tool Use (Final Answer):** If no code is needed, or once the code output is provided, you MUST formulate a clear, policy-relevant, and cited natural language response.
    4. **Citations (Traceability):** For every data-backed claim, you MUST state the data was derived from **IMD Rainfall Data** and/or **Lac Production Data**. Also, **if the question asks for data outside the {st.session_state.get('data_range', '2007-2011')} range, you must state that the required data is not available and answer based only on the available range if possible.**

    Example Code Generation for Reference:
    # Example: Correlation (Using .to_string() instead of .to_markdown())
    corr = df_unified.groupby('State')[['Rainfall_Monsoon_mm', 'Lac_Production_Tons']].corr().unstack().iloc[:,1]['Rainfall_Monsoon_mm']
    print(corr.sort_values(ascending=False).to_string(header=True))

    Begin by strictly following this process for the user's question.
    """

    # 2. Initial Prompt (Ask the LLM to generate code)
    prompt_for_code = f"User Question: '{question}'.\n\nBased on the 'df_unified' schema, generate the Python code required to answer this question. Output ONLY the code block."

    # --- Call 1: Generate Code ---
    
    MAX_RETRIES = 3
    retry_delay = 1
    
    for attempt in range(MAX_RETRIES):
        try:
            headers = {'Content-Type': 'application/json'}
            payload = {
                "contents": [{"parts": [{"text": prompt_for_code}]}],
                "systemInstruction": {"parts": [{"text": system_instruction}]},
            }
            
            # Pass the API key in the URL query parameter
            api_call_url = f"{API_URL}?key={api_key_to_use}"
            response = requests.post(api_call_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            generated_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "")
            break
        except Exception as e:
            st.warning(f"API Call 1 failed (Attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {retry_delay}s. Error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                return f"Error connecting to Gemini API for code generation after {MAX_RETRIES} attempts. Details: {e}"

    # --- 3. Execute Generated Code ---
    code_to_exec = ""
    code_output = "No data analysis was required. Responding conversationally."
    
    st.session_state['code_executed'] = "" 

    try:
        if '```python' in generated_text:
            code_block = generated_text.split('```python')[1].split('```')[0].strip()
            code_to_exec = code_block
            
            exec_globals = {'df_unified': df_unified, 'pd': pd}
            
            # Capture output
            buffer = io.StringIO()
            exec(code_block, exec_globals, {'print': lambda *args, **kwargs: buffer.write(' '.join(map(str, args)) + '\n')})
            code_output = buffer.getvalue().strip()
            st.session_state['code_executed'] = code_to_exec

    except Exception as e:
        code_output = f"Code Execution Error: {e}\n\nGenerated Code:\n{code_to_exec}"

    # --- 4. Call 2: Interpret Result and Formulate Final Answer ---
    
    final_prompt = f"""
    I asked the data analysis agent the question: '{question}'.
    
    The agent generated and executed the following Python code:
    --- CODE START ---
    {code_to_exec}
    --- CODE END ---
    
    The output of the executed code was:
    --- OUTPUT START ---
    {code_output}
    --- OUTPUT END ---
    
    Now, acting as Project Samarth, provide the final, synthesized, and policy-relevant answer to the user. You MUST explicitly state the sources (IMD Rainfall Data and/or Lac Production Data) used for the claim based on the execution result. If the execution resulted in an error, apologize and suggest a rephrase. Ensure the output is formatted as a single, coherent response.
    """

    retry_delay = 1
    for attempt in range(MAX_RETRIES):
        try:
            payload = {
                "contents": [{"parts": [{"text": final_prompt}]}],
                "systemInstruction": {"parts": [{"text": system_instruction}]},
            }
            api_call_url = f"{API_URL}?key={api_key_to_use}"
            response = requests.post(api_call_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            final_answer = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "Could not formulate final answer.")
            break
        except Exception as e:
            st.warning(f"API Call 2 failed (Attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {retry_delay}s. Error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                final_answer = f"Error connecting to Gemini API for interpretation after {MAX_RETRIES} attempts. Details: {e}"
    
    return final_answer

# --- 4. STREAMLIT APPLICATION (UI) ---

def main():
    st.set_page_config(
        page_title="Project Samarth: Cross-Domain Data Q&A",
        layout="wide"
    )

    st.title("🇮🇳 Project Samarth: Cross-Domain Data Q&A Prototype")
    st.markdown(
        """
        **Mission:** Synthesize insights across disparate government datasets (Agriculture & Climate) using a functional LLM-powered Q&A system.
        """
    )

    # --- Data Path Section (Automated Loading) ---
    st.sidebar.header("1. Data Sourcing (Phase 1)")
    st.sidebar.markdown(
        """
        The system is configured to **automatically load data** from the following hardcoded local path:
        """
    )

    # Hardcoded folder path
    folder_path = r"C:\Users\vighn\OneDrive\Desktop\Python\project-samarth\data"
    st.sidebar.info(f"Path: `{folder_path}`")
    
    # Define required filenames
    rf_filename = "RF_SUB_1901-2021.csv"
    lac_filename = "Table_5.14_State-wise_production_of_lac_in_India.csv"
    
    df_unified = None

    # Check if data is already loaded in session state
    if 'df_unified' not in st.session_state:
        # Load and Integrate Data automatically
        try:
            with st.spinner("⏳ Automatically loading and integrating data..."):
                df_unified = preprocess_data(folder_path, rf_filename, lac_filename)
                st.session_state['df_unified'] = df_unified
            
            st.sidebar.success("✅ Datasets Integrated Successfully!")
            st.sidebar.write(f"**Unified Records:** {len(df_unified)} rows available for analysis (Years: {st.session_state.get('data_range', 'N/A')}).")
            
            # Initialize chat messages for the new context
            st.session_state['messages'] = [
                {"role": "assistant", "content": f"Data loaded! We have {len(df_unified)} normalized records (Years: {st.session_state.get('data_range', 'N/A')}). Ask me a question about the correlation between monsoon rainfall and lac production."}
            ]
            # Use st.rerun() to update the chat box immediately after loading the data
            st.rerun()

        except FileNotFoundError as e:
            st.sidebar.error(f"File not found: {e}. Please ensure the hardcoded path and filenames are correct.")
            # Ensure the initial message reflects the loading failure
            st.session_state['messages'] = [
                {"role": "assistant", "content": f"ERROR: Data loading failed. Check the path and files: {e}"}
            ]
        except Exception as e:
            st.sidebar.error(f"Error during data processing: {e}")

    # Check if data is loaded (either from session state or just loaded)
    df_unified = st.session_state.get('df_unified', None)
    if df_unified is not None:
        with st.expander("🔬 View Unified Dataset Schema"):
            st.dataframe(df_unified.head())
            st.info("This is the normalized dataset used by the LLM agent, successfully integrating two disparate sources on 'State' and 'Year'.")


    # --- Chat Interface (Phase 2 Solution) ---

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "assistant", "content": "Welcome to Project Samarth! The system is attempting to load data automatically."}
        ]
    if 'code_executed' not in st.session_state:
        st.session_state['code_executed'] = ""

    # Display chat messages
    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Display the last executed code (for traceability and debugging)
    if st.session_state['code_executed']:
        with st.expander("💻 View Executed Python Code (Traceability)"):
            st.code(st.session_state['code_executed'], language='python')
        

    # Handle user input
    if question := st.chat_input("Ask a policy question about climate and agriculture..."):
        if df_unified is None:
            st.warning("Data could not be loaded. Please check the sidebar for error messages and ensure the hardcoded path is correct.")
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "assistant", "content": "Data is unavailable due to a loading error. Please check the sidebar for details."})
            st.rerun()
            return

        # 1. Add user message to state
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 2. Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Project Samarth is synthesizing data and reasoning..."):
                # Pass the unified DataFrame to the LLM agent logic
                final_answer = answer_question(question, df_unified)
                st.markdown(final_answer)

            # 3. Add assistant response to state
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
            # Use st.rerun() to update the code display expender
            st.rerun()
