import faiss
import numpy as np
import openai
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from sklearn.preprocessing import normalize
import streamlit as st
import io

# Set up OpenAI API key
openai.api_key = st.secrets ["API_KEY"]

# Function to load schema from uploaded CSV
def csv_to_dict(csv_file):
    df = pd.read_csv(csv_file)
    schema = {}
    grouped = df.groupby(['Database Name', 'Table Schema', 'Table Name'])
    for (db_name, schema_name, table_name), group in grouped:
        if db_name not in schema:
            schema[db_name] = {}
        schema[db_name][table_name] = {
            'schema_name': schema_name,
            'columns': {row['Column Name']: row['Data Type'] for _, row in group.iterrows()}
        }
    return schema

# Function to generate embeddings using OpenAIEmbeddings
def generate_embeddings(texts):
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai.api_key)
    embeddings = embeddings_model.embed_documents(texts)
    return embeddings

# Function to chunk schema and create embeddings
def chunk_schema(schema):
    chunks = []
    for db_name, tables in schema.items():
        for table_name, table_data in tables.items():
            columns = table_data["columns"]
            column_details = ", ".join([f"{col} ({dtype})" for col, dtype in columns.items()])
            chunk = f"Database: {db_name}\nTable: {table_name}\nColumns: {column_details}\n"
            chunks.append(chunk)
    return chunks

# Store schema chunks in FAISS
def store_schema_in_faiss(schema_chunks):
    embeddings = generate_embeddings(schema_chunks)
    normalized_embeddings = normalize(np.array(embeddings))
    dimension = len(normalized_embeddings[0])
    index = faiss.IndexFlatIP(dimension)
    index.add(normalized_embeddings)
    return index, schema_chunks

# Query the FAISS index for relevant schema
def query_faiss_index(query, index, schema_chunks, top_k=5):
    query_embedding = generate_embeddings([query])
    query_embedding = normalize(np.array(query_embedding))
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [schema_chunks[idx] for idx in indices[0]]
    return "\n\n".join(relevant_chunks)

# Get the relevant database and tables using LLM
def get_relevant_db_table(user_query, schema_chunk):
    prompt_template = """
    Schema Example:
    Database: sales
    Table: salesorder_history
    Columns: product_id (int), region (varchar), quantity (int), customer_name (varchar)

    User Query: "What is the total quantity of products sold per region by each customer?"
    Response:
    Database: sales
    Table(s): salesorder_history

    Now, based on the following schema chunks:
    {schema_chunk}

    The user has asked:
    "{user_query}"

    Identify:
    - The database name
    - The table(s) required to answer the query

    Provide the response strictly in this format:
    Database: <database_name>
    Table(s): <table_name1>, <table_name2>, ...
    """
    model = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=openai.api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["schema_chunk", "user_query"])
    llm_chain = LLMChain(llm=model, prompt=prompt)
    response = llm_chain.run({"schema_chunk": schema_chunk, "user_query": user_query})
    db_name, table_names = None, None
    try:
        response_lines = response.strip().split("\n")
        db_name = response_lines[0].split("Database:")[-1].strip()
        table_names = response_lines[1].split("Table(s):")[-1].strip()
    except Exception as e:
        st.error(f"Error parsing LLM response: {e}")
        db_name, table_names = "unknown", "unknown"
    return db_name, table_names

# Generate SQL query using LLM
def generate_sql_query(metadata, user_question, db_name):
    prompt_template = """
    You are a highly skilled SQL expert. Using the provided metadata, generate an SQL query that answers the user's question.
    If the query spans multiple tables, include the appropriate JOINs and explain any assumptions.
    Before the SQL query, always include 'USE {db_name};' to specify the database to use.

    Metadata:
    {metadata}

    Question:
    {user_question}

    Database_name:
    {db_name}

    If the question involves aggregating data, ensure to provide clear GROUP BY and aggregation columns.
    If you're unsure about any column names, assume that they exist in the provided metadata.

    Provide the SQL query, followed by a brief explanation of how it answers the question.
    """
    model = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=openai.api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["metadata", "user_question", "db_name"])
    llm_chain = LLMChain(llm=model, prompt=prompt)
    response = llm_chain.run({"metadata": metadata, "user_question": user_question, "db_name": db_name})
    return response.strip()

# Main process function
def process_query(schema, user_question):
    schema_chunks = chunk_schema(schema)
    index, schema_chunks = store_schema_in_faiss(schema_chunks)
    relevant_schema_chunk = query_faiss_index(user_question, index, schema_chunks)
    if not relevant_schema_chunk.strip():
        st.error("FAISS could not find relevant schema chunks.")
        return None
    db_name, table_names = get_relevant_db_table(user_question, relevant_schema_chunk)
    print(f"Identified Database: {db_name}, Table(s): {table_names}")
    try:
        metadata = {table_name: schema[db_name][table_name] for table_name in table_names.split(", ")}
    except KeyError as e:
        st.error(f"Error finding metadata: {e}")
        return None
    sql_query = generate_sql_query(metadata, user_question, db_name)
    return sql_query

# Streamlit main function
def main():
    st.set_page_config("SQL Query Automation")
    st.header("SQL Query Automation")
    with st.sidebar:
        st.title("About the Tool")
        st.write("Upload your CSV file and ask your SQL-related questions.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for i in st.session_state.chat_history:
        st.chat_message(i["role"]).write(i["content"])
    uploaded_file = st.sidebar.file_uploader("Upload your CSV schema file", type=["csv"])
    if uploaded_file is not None:
        schema = csv_to_dict(uploaded_file)
        st.sidebar.success("Schema loaded successfully!")
        user_question = st.chat_input("Enter your SQL question...")
        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.chat_message("user").write(user_question)
            with st.spinner("Processing..."):
                answer = process_query(schema, user_question)
                if answer:
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.chat_message("assistant").write(answer)
                else:
                    st.error("No valid response generated.")
                    
main()
