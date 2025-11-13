import os


def read_schema(schema_path):
    '''
    Read the .schema file
    '''
    with open(schema_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_sql_query(response):
    '''
    Extract the SQL query from the model's response
    '''
    text = response

    # 1) Try fenced ```sql ... ```
    m = re.search(r"```(?:sql)?\s*(.*?)```", text,
                  flags=re.IGNORECASE | re.DOTALL)
    if m:
        candidate = m.group(1)
    else:
        candidate = text

    # 2) Find the first 'SELECT ... ;'
    m2 = re.search(r"(SELECT[\s\S]*?;)", candidate, flags=re.IGNORECASE)
    if m2:
        query = m2.group(1)
    else:
        # fallback: take line that contains SELECT
        lines = candidate.splitlines()
        sel_lines = [ln for ln in lines if "select" in ln.lower()]
        if sel_lines:
            # if no semicolon, add one
            query = sel_lines[0].strip()
            if not query.endswith(";"):
                query += ";"
        else:
            # give up: return as-is (evaluator will mark empty/non-executable)
            query = candidate.strip()

    # Cleanup extra formatting or prefix
    query = query.strip()
    # Remove possible "SQL:" prefixes etc
    query = re.sub(r"^(SQL\s*:\s*)", "", query, flags=re.IGNORECASE).strip()
    return query


def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"SQL EM: {sql_em}\n")
        f.write(f"Record EM: {record_em}\n")
        f.write(f"Record F1: {record_f1}\n")
        f.write("Model Error Messages:\n")
        for i, em in enumerate(error_msgs or []):
            f.write(f"[{i}] {em}\n")
