import sqlite3
import os

# 1. Use absolute path to ensure we hit the right file
db_path = os.path.join(os.getcwd(), 'mlflow.db')
print(f"Targeting database at: {db_path}")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

old_p = "C:/Users/himan/Documents/Projects/ml-risk-prediction"
new_p = "/app"

updates = [
    ("runs", "artifact_uri"),
    ("experiments", "artifact_location"),
    ("model_versions", "storage_location")
]

for table, col in updates:
    # Perform the update
    cursor.execute(f"UPDATE {table} SET {col} = REPLACE({col}, '{old_p}', '{new_p}')")
    print(f"Updated {cursor.rowcount} rows in {table}")

# 2. MANDATORY: Commit the changes to the disk
conn.commit()

# 3. VERIFICATION: Check one row to see if it actually changed
cursor.execute("SELECT artifact_uri FROM runs LIMIT 1")
row = cursor.fetchone()
if row:
    print(f"Verification - Current path in DB: {row[0]}")
    if "/app" in row[0]:
        print("✅ SUCCESS: Paths are now in Linux format.")
    else:
        print("❌ FAILURE: Paths are still in Windows format.")

conn.close()