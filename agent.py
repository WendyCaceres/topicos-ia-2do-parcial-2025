import dspy
import sqlite3
from dotenv import load_dotenv

from tools import execute_sql, get_schema, save_data_to_csv


# --- DSPy Agent Definition ---
class SQLAgentSignature(dspy.Signature):
    """
        Agente ReAct para consultar bases de datos SQLite usando lenguaje natural.

        Responsabilidades:
        - Comprender la solicitud del usuario en lenguaje natural.
        - Usar las herramientas para inspeccionar la estructura de la base de datos y ejecutar sentencias SQL.
        - Interpretar los resultados y proporcionar una explicación clara y concisa al usuario.

        Herramientas disponibles:
        - execute_sql: ejecuta sentencias SQL (SELECT, INSERT, UPDATE, DELETE).
        - get_schema: devuelve metadatos de tablas y columnas. Usar antes de escribir consultas si el esquema es desconocido.
        - save_data_to_csv: exporta resultados a CSV cuando se solicite.

        Operaciones permitidas:
        - READ: consultas SELECT.
        - CREATE: inserciones con INSERT.
        - UPDATE: modificaciones con UPDATE.
        - DELETE: eliminaciones con DELETE.

        Directrices:
        - Si hay dudas sobre nombres de tablas o columnas, usar get_schema primero.
        - En caso de error en execute_sql, analizar el mensaje y corregir la consulta.
        - Antes de INSERT o UPDATE, verificar nombres y tipos de columnas con el esquema.
        - Tras INSERT, UPDATE o DELETE, ejecutar un SELECT para confirmar el cambio cuando sea apropiado.
        - Usar save_data_to_csv para exportar resultados.
        - Proporcionar respuestas claras y breves en lenguaje natural.
        - Ser especialmente cuidadoso con DELETE: confirmar qué se eliminará antes de ejecutar.
        - Iterar para corregir errores, manteniendo las llamadas a herramientas eficientes.
    """

    question = dspy.InputField(desc="La pregunta en lenguaje natural del usuario.")
    initial_schema = dspy.InputField(desc="El esquema inicial de la base de datos para guiarte.")
    answer = dspy.OutputField(
        desc="La respuesta final en lenguaje natural a la pregunta del usuario."
    )


class SQLAgent(dspy.Module):
    """The SQL Agent Module"""
    def __init__(self, tools: list[dspy.Tool]):
        super().__init__()
        # Initialize the ReAct agent.
        self.agent = dspy.ReAct(
            SQLAgentSignature,
            tools=tools,
            max_iters=7,  # Set a max number of steps
        )

    def forward(self, question: str, initial_schema: str) -> dspy.Prediction:
        """The forward pass of the module."""
        result = self.agent(question=question, initial_schema=initial_schema)
        return result


def configure_llm():
    """Configures the DSPy language model."""
    load_dotenv()
    llm = dspy.LM(model="openai/gpt-4o-mini", max_tokens=4000)
    dspy.settings.configure(lm=llm)

    print("[Agent] DSPy configured with gpt-4o-mini model.")
    return llm


def create_agent(conn: sqlite3.Connection, query_history: list[str] | None = None) -> dspy.Module | None:
    if not configure_llm():
        return

    execute_sql_tool = dspy.Tool(
        name="execute_sql",
        # ===> (1.1.2) YOUR execute_sql TOOL DESCRIPTION HERE
        desc=(
            "Ejecuta una consulta SQL en la base de datos SQLite."
            "Input: consulta (str, sentencia SQL valida como ser SELECT, INSERT, UPDATE, DELETE). "
            "Output: Representación en cadena de los resultados como una lista de tuplas (por ejemplo, '[(1, \"Alice\"), (2, \"Bob\")]') "
            "para consultas SELECT, o un mensaje de éxito/error para otras operaciones."
        ),
        # Use lambda to pass the 'conn' object
        func=lambda query: execute_sql(conn, query, query_history),
    )

    get_schema_tool = dspy.Tool(
        name="get_schema",
        # ===> (1.1.2) YOUR get_schema_tool TOOL DESCRIPTION HERE
        desc=(
            "Devuelve el esquema de la base de datos para una tabla específica o todas las tablas. "
            "Input: table_name (str o None). Si es None, devuelve una lista de todos los nombres de las tablas. "
            "Si se proporciona un nombre de tabla, devuelve una cadena con los nombres y tipos de columnas para esa tabla "
            "(por ejemplo, '[('id', 'INTEGER'), ('name', 'TEXT')]')."
        ),
        # Use lambda to pass the 'conn' object
        func=lambda table_name: get_schema(conn, table_name),
    )

    save_csv_tool = dspy.Tool(
        name="save_data_to_csv",
        # ===> YOUR save_csv_tool TOOL DESCRIPTION HERE
        desc=(
            "Guarda los resultados de una consulta tabular en un archivo CSV. "
            "Inputs: "
            "  - data: lista de tuplas/listas O el string devuelto por execute_sql "
            "(por ejemplo, '[(1, \"Alice\"), (2, \"Bob\")]'). "
            "  - filename: nombre de archivo de salida deseado (str, .csv extension added automatically). "
            "Output: Mensaje de éxito con la ruta absoluta del archivo o descripción del error."
        ),
        func=save_data_to_csv
    )

    all_tools = [execute_sql_tool, get_schema_tool, save_csv_tool]     # Add save_csv_tool when completed
    all_tools = [execute_sql_tool, get_schema_tool, save_csv_tool]  
    # 2. Instantiate and run the agent
    agent = SQLAgent(tools=all_tools)

    return agent
