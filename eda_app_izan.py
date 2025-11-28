# *Assigment 1 : Streamlit*

# Design a Data Science EDA App

# a.) user can upload csv , excel 
# b.) basic eda --> show preview , info , describe , no. of missing values , no. of duplicate records
# c.) ask user to select columns [multiselect]
# d.) provide some diff diff graphs to the user
# e.) generate the graph using seaborn + matplotlib
# f.) ask user for some query
#     user_query : show me top 5 categories 
#	  result     : dataframe
	  
#	  user_query : show me those records where customer initiated more than 5 customer service calls

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
def configure_page():
    """Sets up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Advanced EDA App with Filtering",
        layout="wide",
        initial_sidebar_state="expanded"
    )

@st.cache_data
def load_data(uploaded_file):
    """Loads data based on file type (CSV or Excel)."""
    if uploaded_file.name.endswith('.csv'):
        # Added on_bad_lines='skip' to handle potentially malformed CSVs
        df = pd.read_csv(uploaded_file, on_bad_lines='skip') 
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format.")
        return None
    return df

# --- Filtering Function ---
def interactive_filter(df):
    """Creates sidebar widgets for filtering the DataFrame."""
    st.sidebar.markdown("---")
    st.sidebar.header("Data Filter ⚙️")

    # Start with the original dataframe
    df_filtered = df.copy()

    # Get a list of numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.sidebar.subheader("Numerical Filters")
    # Apply filters for each numerical column
    for col in numerical_cols:
        # Determine min/max dynamically from the original data
        min_val = df[col].min()
        max_val = df[col].max()
        
        # Create a slider for the user to select a range
        col_range = st.sidebar.slider(
            f'Filter {col}',
            float(min_val), # Convert to float for slider compatibility
            float(max_val), 
            (float(min_val), float(max_val)) # Default to full range
        )
        # Apply the filter condition
        df_filtered = df_filtered[
            (df_filtered[col] >= col_range[0]) & 
            (df_filtered[col] <= col_range[1])
        ]

    st.sidebar.subheader("Categorical Filters")
    # Apply filters for each categorical column
    for col in categorical_cols:
        unique_values = df[col].unique().tolist()
        
        # Create a multiselect for the user to choose values
        selected_values = st.sidebar.multiselect(
            f'Filter {col}',
            options=unique_values,
            default=unique_values # Default to all values selected
        )
        
        # Apply the filter condition using .isin()
        if selected_values:
            df_filtered = df_filtered[df_filtered[col].isin(selected_values)]
            
    return df_filtered

# --- Main Application Logic ---
def main():
    configure_page()

    st.title("Pandas & Streamlit: Advanced EDA Application 📈")
    st.markdown("Upload a **CSV** or **Excel** file to perform comprehensive EDA with **interactive filtering**.")

    # 1. File Uploader
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV or Excel file",
        type=["csv", "xls", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            df_original = load_data(uploaded_file)
            if df_original is None:
                return

            st.sidebar.success("File loaded successfully!")
            
            # 2. Interactive Filtering (NEW STEP)
            df = interactive_filter(df_original.copy())

            # 3. Display Filter Status
            st.header("1. Filtered Dataset Status")
            st.info(f"The displayed analysis is based on a filtered dataset containing **{len(df)}** out of {len(df_original)} original records.")
            
            if df.empty:
                st.error("The current filter settings resulted in an empty dataset. Adjust your filters in the sidebar.")
                return # Stop further analysis if no data remains

            # 4. Basic EDA Metrics and Preview (b) - Uses filtered df
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Filtered Rows", value=df.shape[0])
            with col2:
                st.metric(label="Columns", value=df.shape[1])
            with col3:
                num_duplicates = df.duplicated().sum()
                st.metric(label="Duplicate Records (Filtered)", value=num_duplicates)

            # --- Missing Values (Filtered) ---
            st.subheader("Missing Values Count")
            missing_data = df.isnull().sum().sort_values(ascending=False)
            missing_data = missing_data[missing_data > 0]
            
            if not missing_data.empty:
                st.dataframe(missing_data.rename('Missing Count').to_frame())
            else:
                st.info("No missing values found in the current filtered dataset.")

            # --- Data Preview (Head) ---
            st.subheader("Data Preview (First 5 Rows)")
            st.dataframe(df.head())

            # --- Data Info (d-types) ---
            st.subheader("Column Information (d-types)")
            buffer = pd.io.common.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

            # --- Descriptive Statistics ---
            st.subheader("Descriptive Statistics")
            st.dataframe(df.describe().T)

            # --- Visualization Section (c, d, e) ---
            st.header("2. Data Visualization")

            all_cols = df.columns.tolist()
            selected_cols = st.multiselect(
                "Select one or more columns for plotting:",
                options=all_cols,
                default=all_cols[0] if all_cols else []
            )

            # Rest of the visualization logic (using df_filtered)
            # ... (Existing plotting code goes here)
            # --- START PLOTTING LOGIC ---
            if selected_cols:
                st.subheader(f"Visualization for Selected Columns: {', '.join(selected_cols)}")
                
                # Determine available graph types based on selection
                if len(selected_cols) == 1:
                    col = selected_cols[0]
                    if pd.api.types.is_numeric_dtype(df[col]):
                        graph_options = ["Histogram", "Box Plot"]
                    elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                        graph_options = ["Count Plot"]
                    else:
                        st.info("Selected column is not suitable for basic univariate plotting.")
                        return

                elif len(selected_cols) == 2:
                    col_x, col_y = selected_cols[0], selected_cols[1]
                    if pd.api.types.is_numeric_dtype(df[col_x]) and pd.api.types.is_numeric_dtype(df[col_y]):
                        graph_options = ["Scatter Plot", "Correlation Plot"]
                    else:
                        graph_options = []
                        st.warning("For bivariate plots, please select two numerical columns.")
                else:
                    st.warning("Please select 1 or 2 columns to generate a standard plot.")
                    graph_options = []


                if graph_options:
                    plot_type = st.selectbox("Select Plot Type:", graph_options)

                    # Generate the plot
                    fig, ax = plt.subplots(figsize=(10, 6))

                    if plot_type == "Histogram":
                        sns.histplot(df[col], kde=True, ax=ax, bins=30)
                        ax.set_title(f"Histogram of {col}")
                        ax.set_xlabel(col)
                        
                    elif plot_type == "Box Plot":
                        sns.boxplot(y=df[col], ax=ax)
                        ax.set_title(f"Box Plot of {col} (Outlier Detection)")
                        ax.set_ylabel(col)

                    elif plot_type == "Count Plot":
                        sns.countplot(y=df[col], order=df[col].value_counts().index, ax=ax)
                        ax.set_title(f"Count Plot of {col}")
                        ax.set_xlabel("Count")
                        ax.set_ylabel(col)

                    elif plot_type == "Scatter Plot":
                        sns.scatterplot(x=df[col_x], y=df[col_y], ax=ax)
                        ax.set_title(f"Scatter Plot: {col_x} vs {col_y}")

                    elif plot_type == "Correlation Plot":
                        # Calculate correlation matrix only for numerical columns in selection
                        corr_df = df[[col_x, col_y]].corr()
                        sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                        ax.set_title(f"Correlation Heatmap of {col_x} and {col_y}")


                    st.pyplot(fig)
            # --- END PLOTTING LOGIC ---
                
            # 6. Selection for user query (f) - Uses filtered df
            st.header("3. Data Query Tool")
            user_query = st.text_input(
                "Enter a simple query (e.g., 'df[df[\"Age\"] > 30].head()'):",
                value="df.head()"
            )

            if user_query:
                try:
                    result = eval(user_query, {'pd': pd, 'df': df})
                    
                    st.subheader("Query Result")
                    if isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
                        st.dataframe(result)
                    else:
                        st.code(result)
                except Exception as e:
                    st.error(f"Error executing query: {e}")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            st.info("Please ensure your file is correctly formatted.")

    else:
        st.info("Please upload a CSV or Excel file via the sidebar to begin the EDA.")

# Run the main function
if __name__ == "__main__":
    main()