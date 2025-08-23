''' 
To run without moving the hyperloglog folder, use:
    python -m examples.basic_usage
'''

import psycopg2
from hyperloglog.core import HyperLogLog
from hyperloglog.serialization import serialize_hll, deserialize_hll


# Example datasets with some duplicate values
fruits = ["apple", "banana", "orange", "grape", "mango", "banana", "apple"]
nuts = ["almond", "cashew", "walnut", "almond", "pistachio", "pecan", "walnut"]
animals = ["cat", "dog", "elephant", "tiger", "lion", "dog", "cat"]
people = ["alice", "bob", "charlie", "alice", "dave", "eve", "bob"]

datasets = {
    "fruits": fruits,
    "nuts": nuts,
    "animals": animals,
    "people": people
}


# ------------------ DATABASE CONNECTION ------------------
try:
    conn = psycopg2.connect(
        dbname="test_hll",   # change this 
        user="postgres",     # change this
        password="AIML",     # change this
        host="localhost",
        port=5432
    )
    cur = conn.cursor()
except psycopg2.Error as e:
    print(f"Failed to connect to database: {e}")
    raise


# ------------------ CREATE TABLE ------------------
try:
    # Drop existing table if it exists (for a clean run)
    cur.execute("DROP TABLE IF EXISTS hll_counters")

    cur.execute("""
        CREATE TABLE hll_counters (
            label TEXT,
            hll_data TEXT
        )
    """)
    conn.commit()
    print("Table created successfully.")
except psycopg2.Error as e:
    print(f"Failed to create table: {e}")
    raise


# ------------------ STORE HLLs IN DATABASE ------------------
for label, items in datasets.items():
    # Initialize a new HLL with precision b=14
    hll = HyperLogLog(b=14)

    # Add items 
    for item in items:
        hll.add(item)

    # Estimate cardinality before storing
    estimate_before = hll.estimate()
    print(f"\n[{label.upper()}] Estimate before storing: {estimate_before}")
    print("Registers before:", hll.impl.registers[:10], "...")  # show first 10 registers

    try:
        # Serialize HLL object to base64 string
        b64_data = serialize_hll(hll)

        # Insert into database
        cur.execute(
            "INSERT INTO hll_counters (label, hll_data) VALUES (%s, %s)",
            (label, b64_data)
        )
        conn.commit()
        print(f"Stored HLL for {label} in DB.")
    except psycopg2.Error as e:
        print(f"Failed to insert {label}: {e}")
        raise


# ------------------ FETCH & VERIFY HLLs -------------
for label in datasets.keys():
    try:
        cur.execute("SELECT hll_data FROM hll_counters WHERE label = %s", (label,))
        row = cur.fetchone()

        if row is None:
            raise ValueError(f"No data found for label '{label}'")

        b64_data_fetched = row[0]
        print(f"\n[{label.upper()}] Fetched base64 data (first 50 chars): {b64_data_fetched[:50]}...")

        hll_fetched = deserialize_hll(b64_data_fetched)

        estimate_after = hll_fetched.estimate()
        print(f"Estimate after fetching from DB: {estimate_after}")

        # Warn if estimates differ
        if abs(estimate_before - estimate_after) > 0.001:
            print("Warning: Estimate changed after DB round-trip!")
    except (psycopg2.Error, ValueError, Exception) as e:
        print(f"Error for {label}: {e}")


# ------------------ CLEANUP --------------
try:
    cur.close()
    conn.close()
    print("\nDatabase connection closed.")
except psycopg2.Error as e:
    print(f"Failed to close DB connection: {e}")
    raise
