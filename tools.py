import os
import sqlite3
import csv


def execute_sql(conn: sqlite3.Connection, query: str, query_history: list[str] | None = None) -> str:
    """
    Executes a SQL query.
    Returns the fetched data as a string or the error message as a string.
    """
    print(f"   [BEGIN Tool Action] Executing SQL: {query} [END Tool Action]")
    if query_history is not None:
        query_history.append(query)
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        # Check if it was a query that returns data (SELECT)
        if cursor.description:
            rows = cursor.fetchall()
            return str(rows)  # Return data as a string
        else:
            conn.commit()
            return "Query executed successfully (no data returned)."
    except sqlite3.Error as e:
        print(f"   [ERROR] {e}")
        return f"Error: {e}"  # Return the error message string


def get_schema(conn: sqlite3.Connection, table_name: str | None = None) -> str:
    """
    Gets the schema for all tables or a specific table.
    """
    print(f"   [Tool Action] Getting schema for: {table_name or 'all tables'}")
    cursor = conn.cursor()
    if table_name:
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        # (cid, name, type, notnull, default_value, pk)
        return str([(col[1], col[2]) for col in columns])
    else:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        return str([table[0] for table in tables])


def save_data_to_csv(data: list[tuple] | str, filename: str) -> str:
    """
    Saves query results to a CSV file.
    """
    print(f"   [Tool Action] Saving data to {filename}...")
    try:
        
        if not isinstance(data, list):
            return "Error: Data must be a list of tuples or lists."
        
        rows = [list(row) if isinstance(row, (list, tuple)) else [row] for row in data]
        
        # Ensure .csv extension
        if not filename.lower().endswith(".csv"):
            filename += ".csv"
        
        base_dir = os.path.join(os.getcwd(), "files")
        os.makedirs(base_dir, exist_ok=True)
        file_path = os.path.join(base_dir, filename)
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        abs_path = os.path.abspath(file_path)
        return f"Data saved successfully to: {abs_path}"
    
    except Exception as e:
        return f"Error saving CSV: {str(e)}"