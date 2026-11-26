"""SQLite store with a small, relative-dated demo dataset (so demos never go stale)."""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    status TEXT NOT NULL,
    total REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'EUR',
    items TEXT NOT NULL,
    ordered_on TEXT NOT NULL,
    delivered_on TEXT
);
CREATE TABLE IF NOT EXISTS refunds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL REFERENCES orders(id),
    amount REAL NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,
    created_on TEXT NOT NULL,
    decided_by TEXT
);
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    order_id TEXT,
    summary TEXT NOT NULL,
    created_on TEXT NOT NULL
);
"""

CUSTOMERS = [
    ("C001", "Anna Becker", "anna.becker@example.com"),
    ("C002", "Ravi Kumar", "ravi.kumar@example.com"),
    ("C003", "Lena Schmidt", "lena.schmidt@example.com"),
]

# (id, customer, status, total, items, ordered N days ago, delivered N days ago or None)
ORDERS = [
    ("ORD-1001", "C001", "delivered", 29.90, "Wireless mouse", 14, 8),
    ("ORD-1002", "C001", "delivered", 149.00, "Office chair cushion set", 18, 11),
    ("ORD-1003", "C001", "delivered", 39.00, "USB-C hub", 100, 95),
    ("ORD-1004", "C002", "shipped", 59.00, "Desk lamp", 3, None),
    ("ORD-1005", "C002", "delivered", 19.99, "Phone case", 9, 4),
    ("ORD-1006", "C003", "processing", 89.00, "Keyboard", 1, None),
    ("ORD-1007", "C003", "delivered", 12.50, "Notebook set", 12, 6),
]


def connect(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection, today: date) -> None:
    conn.executescript(SCHEMA)
    if conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]:
        return
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?)", CUSTOMERS)
    for oid, cid, status, total, items, ordered, delivered in ORDERS:
        conn.execute(
            "INSERT INTO orders (id, customer_id, status, total, items, ordered_on, delivered_on)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                oid,
                cid,
                status,
                total,
                items,
                (today - timedelta(days=ordered)).isoformat(),
                (today - timedelta(days=delivered)).isoformat() if delivered is not None else None,
            ),
        )
    conn.commit()


def add_refund(conn: sqlite3.Connection, order_id: str, amount: float, reason: str,
               status: str, today: date) -> int:
    cur = conn.execute(
        "INSERT INTO refunds (order_id, amount, reason, status, created_on) VALUES (?, ?, ?, ?, ?)",
        (order_id, amount, reason, status, today.isoformat()),
    )
    conn.commit()
    return int(cur.lastrowid)
