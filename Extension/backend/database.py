"""
database.py — SQLite detection log for the Cyberbully Shield dashboard.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'detections.db')


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the detections table if it doesn't exist."""
    conn = get_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            content_type TEXT NOT NULL DEFAULT 'text',
            text_snippet TEXT,
            source_url TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            was_censored INTEGER NOT NULL DEFAULT 1
        )
    ''')
    conn.commit()
    conn.close()


def log_detection(text_snippet: str, source_url: str, prediction: str,
                  confidence: float, content_type: str = 'text',
                  was_censored: bool = True):
    """Insert a detection event into the log."""
    conn = get_connection()
    conn.execute(
        '''INSERT INTO detections
           (timestamp, content_type, text_snippet, source_url, prediction, confidence, was_censored)
           VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (datetime.now().isoformat(), content_type,
         text_snippet[:500] if text_snippet else '',
         source_url or '', prediction, confidence, int(was_censored))
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 100):
    """Return the most recent detections."""
    conn = get_connection()
    rows = conn.execute(
        'SELECT * FROM detections ORDER BY id DESC LIMIT ?', (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    """Return aggregate statistics."""
    conn = get_connection()
    total = conn.execute('SELECT COUNT(*) as c FROM detections').fetchone()['c']
    threats = conn.execute(
        "SELECT COUNT(*) as c FROM detections WHERE prediction = 'Cyberbullying'"
    ).fetchone()['c']
    safe = total - threats

    # Detections per hour (last 24h)
    hourly = conn.execute('''
        SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
        FROM detections
        WHERE timestamp >= datetime('now', '-24 hours')
        GROUP BY hour ORDER BY hour
    ''').fetchall()

    # Top source domains
    domains = conn.execute('''
        SELECT source_url, COUNT(*) as count
        FROM detections
        WHERE prediction = 'Cyberbullying'
        GROUP BY source_url ORDER BY count DESC LIMIT 10
    ''').fetchall()

    conn.close()
    return {
        "total_scans": total,
        "threats_detected": threats,
        "safe_content": safe,
        "threat_rate": round(threats / max(total, 1) * 100, 2),
        "hourly": [dict(r) for r in hourly],
        "top_sources": [dict(r) for r in domains],
    }


def clear_history():
    """Clear all detection history."""
    conn = get_connection()
    conn.execute('DELETE FROM detections')
    conn.commit()
    conn.close()
