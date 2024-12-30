import faiss
from langchain_community.vectorstores import FAISS
import numpy as np
import openai
import pyodbc
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from sklearn.preprocessing import normalize
import streamlit as st

# Set up OpenAI API key
# server = st.secrets["server"]
# database = st.secrets["database"]
# user = st.secrets["user"]
# password = st.secrets["password"]
# API_KEY = st.secrets["API_KEY"]
openai.api_key = API_KEY

# Function to connect to MS SQL Server and fetch schema
def fetch_schema_from_mssql(server, database, user, password):
    conn = pyodbc.connect(f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={user};PWD={password};Timeout=60')
    cursor = conn.cursor()

    schema = {}

    # Fetch all databases
    cursor.execute("SELECT name FROM sys.databases WHERE state_desc = 'ONLINE'")
    databases = cursor.fetchall()

    for db in databases:
        db_name = db[0]
        schema[db_name] = {}

        # Switch to the database
        cursor.execute(f"USE {db_name}")

        # Fetch all tables and their columns
        cursor.execute("""
            SELECT TABLE_SCHEMA, TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE'
        """)
        tables = cursor.fetchall()

        for table in tables:
            table_schema = table[0]
            table_name = table[1]

            # Fetch columns for each table
            cursor.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = '{table_schema}' AND TABLE_NAME = '{table_name}'
            """)
            columns = cursor.fetchall()

            schema[db_name][table_name] = {
                'schema_name': table_schema,
                'columns': {col[0]: col[1] for col in columns}
            }

    conn.close()
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

    # Create FAISS index for cosine similarity
    dimension = len(normalized_embeddings[0])  # Length of embedding vectors
    index = faiss.IndexFlatIP(dimension)  # Cosine similarity via Inner Product

    # Add embeddings to the FAISS index
    index.add(normalized_embeddings)

    return index, schema_chunks

# Query the FAISS index for relevant schema
def query_faiss_index(query, index, schema_chunks, top_k=3):
    query_embedding = generate_embeddings([query])
    query_embedding = normalize(np.array(query_embedding))

    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve top-K schema chunks
    relevant_chunks = [schema_chunks[idx] for idx in indices[0]]
    return "\n\n".join(relevant_chunks)  # Combine multiple chunks for context

def get_relevant_db_table(user_query, schema_chunk):
    prompt_template = """
    Given the following schema chunks:
    {schema_chunk}

    User Query: "{user_query}"

    Identify the most relevant database and table(s) to answer the query. If the query requires columns from multiple tables, list all tables. Provide the result strictly in this format:
    Database: <database_name>
    Table(s): <table_name1>, <table_name2>, ...
    """

    # Initialize the ChatOpenAI model
    model = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=openai.api_key)

    # Create a prompt template
    prompt = PromptTemplate(template=prompt_template, input_variables=["schema_chunk", "user_query"])

    # Run the LLM chain
    llm_chain = LLMChain(llm=model, prompt=prompt)
    response = llm_chain.run({"schema_chunk": schema_chunk, "user_query": user_query})

    # Print the response for debugging (optional)
    print(f"LLM Response: {response}")

    # Parse the response
    response_lines = response.strip().split("\n")
    if len(response_lines) > 1:
        db_name = response_lines[0].replace("Database: ", "").strip()
        table_names = response_lines[1].replace("Table(s): ", "").strip()
    else:
        st.error("Response does not contain the expected table information.")

    

    return db_name, table_names

# Function to generate SQL query using LLM
def generate_sql_query(metadata, user_question, db_name):
    prompt_template = """
    Using the provided metadata, generate an SQL query that answers the user's question.
    If the query spans multiple tables, include appropriate JOINs. Before the SQL query, always include 'USE {db_name};' in the query.

    Metadata:
    {metadata}

    Question:
    {user_question}

    Database_name:
    {db_name}

    Provide the SQL query and a brief explanation.
    """
    model = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=openai.api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["metadata", "user_question", "db_name"])
    llm_chain = LLMChain(llm=model, prompt=prompt)

    # Run the chain with the provided metadata and user question
    response = llm_chain.run({"metadata": metadata, "user_question": user_question, "db_name": db_name})
    return response.strip()


# Main function to process user query and generate SQL query
def process_query(schema , user_question ):

    # Step 2: Chunk the schema and store in FAISS
    schema_chunks = chunk_schema(schema)
    index, schema_chunks = store_schema_in_faiss(schema_chunks)

    # Step 3: Get the relevant schema chunk using FAISS
    relevant_schema_chunk = query_faiss_index(user_question, index, schema_chunks)

    # Step 4: Get the relevant database and table using LLM
    db_name, table_names = get_relevant_db_table(user_question, relevant_schema_chunk)
    print(f"Identified Database: {db_name}, Table(s): {table_names}")

    # Step 5: Get the schema of the identified table(s)
    metadata = {table_name: schema[db_name][table_name] for table_name in table_names.split(", ")}

    # Step 6: Generate SQL query using LLM
    sql_query = generate_sql_query(metadata, user_question,db_name)

    # Step 7: Return the generated SQL query (no execution)
    return sql_query

def main():
    st.set_page_config("Sql Query Automation")
    st.header("Sql Query Automation")
    with st.sidebar:
      st.title("About the Tool")
      st.write("This tool helps you generate SQL queries effortlessly. Enter your question, and let the system create the corresponding SQL query for you.")
      st.write("Steps to Use:")
      st.write("1. Enter your natural language question.")
      st.write("2. Review the generated SQL query and explanation.")

    
    if "chat_history" not in st.session_state:      # creating chat history
        st.session_state.chat_history = []

    for i in st.session_state.chat_history:          # iterating to chat_histroy to display all the chat
        st.chat_message(i["role"]).write(i["content"])

    schema = fetch_schema_from_mssql(server, database, user, password)
    st.sidebar.success("Fetched Schema Sucessfully!")
    user_question = st.chat_input("Enter your Question...")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})  # appending user message to the chat_history
        st.chat_message("user").write(user_question)

        with st.spinner("Generating response..."):
        
            answer = process_query(schema, user_question)
            if answer is not None:
                response = answer
            else:
                st.error("No relevant answer found in the uploaded documents.")


        st.session_state.chat_history.append({"role": "assistant", "content": response})  # Appending bot response to the chat history in session_state 
        st.chat_message("assistant").write(response)

    
main()
