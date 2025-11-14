import os
import re


def read_schema(schema_path):
    '''
    Read the .schema file
    '''
    # Minimal parser: return raw text and simple table->columns mapping if present
    schema = {"raw": "", "tables": {}}
    if not os.path.exists(schema_path):
        return schema
    with open(schema_path, "r", encoding="utf-8") as f:
        txt = f.read()
    schema["raw"] = txt
    # Try to parse lines like: table_name(col1, col2, ...)
    for line in txt.splitlines():
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*$", line)
        if m:
            tbl = m.group(1)
            cols = [c.strip() for c in m.group(2).split(",") if c.strip()]
            schema["tables"][tbl] = cols
    return schema


def extract_sql_query(response):
    '''
    Extract the SQL query from the model's response
    '''
    # Heuristic: take substring from first "select" (case-insensitive) to the end of the first ';'
    s = response.strip()
    m = re.search(r"(select\s.+?;)", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: return the line that starts with SELECT if any
    for ln in s.splitlines():
        if re.match(r"\s*select\s", ln, flags=re.IGNORECASE):
            return ln.strip()
    return s.strip()


def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(
            f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")
