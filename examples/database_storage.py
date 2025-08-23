'''
To run without moving the hyperloglog folder, use:
    python -m examples.database_storage
'''

import psycopg2
from hyperloglog.core import HyperLogLog
from hyperloglog.serialization import serialize_hll, deserialize_hll


# ------------------ DATABASE CONNECTION ------------------
try:
    conn = psycopg2.connect(
        dbname="test_hll",   # change 
        user="postgres",     # change 
        password="AIML",     # change 
        host="localhost",    
        port=5432            
    )
    cur = conn.cursor()
    print("Connected to database.")
except psycopg2.Error as e:
    print(f"Failed to connect to DB: {e}")
    raise


# ------------------ CREATE TABLE ------------------
try:
    # Drop table if it already exists (for a clean run)
    cur.execute("DROP TABLE IF EXISTS hll_counters")

    cur.execute("""
        CREATE TABLE hll_counters (
            label TEXT,
            hll_data TEXT
        )
    """)
    conn.commit()
    print("Table hll_counters created.")
except psycopg2.Error as e:
    print(f"Failed to create table: {e}")
    raise


# ------------------ CREATE & POPULATE HLL ------------------
hll = HyperLogLog(b=14)

# Add 50,000 unique items into the HLL
for i in range(50_000):
    hll.add(f"item{i}")

estimate = hll.estimate()
print(f"\nEstimate from HLL: {estimate}")
print(f"Actual count: {50_000} (for comparison)")


# ------------------ (Optional) STORE IN DB ------------------
try:
    # Serialize HLL object to base64 string
    b64_data = serialize_hll(hll)

    cur.execute(
        "INSERT INTO hll_counters (label, hll_data) VALUES (%s, %s)",
        ("test_hll", b64_data)
    )
    conn.commit()
    print("HLL stored in database as 'test_hll'.")
except psycopg2.Error as e:
    print(f"Failed to insert HLL: {e}")
    raise


# ------------------ (Optional) FETCH BACK & VERIFY ------------------
try:
    cur.execute("SELECT hll_data FROM hll_counters WHERE label = %s", ("test_hll",))
    row = cur.fetchone()

    if row:
        # Deserialize string back into HLL object
        hll_fetched = deserialize_hll(row[0])
        estimate_fetched = hll_fetched.estimate()

        print(f"\nEstimate after fetching from DB: {estimate_fetched}")
    else:
        print("No HLL data found in DB.")
except psycopg2.Error as e:
    print(f"Failed to fetch HLL: {e}")


# ------------------ CLEANUP ------------------
try:
    cur.close()
    conn.close()
    print("\nDatabase connection closed.")
except psycopg2.Error as e:
    print(f"Failed to close DB connection: {e}")
    raise
