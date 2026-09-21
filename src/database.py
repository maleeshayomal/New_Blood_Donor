"""
SQLite Database Helper Module
==============================
Academic AI Decision-Support Prototype

Manages structured relational storage for donors, emergency blood requests,
donation logs, and recommendation predictions.
"""

import sqlite3
import os
from typing import List, Dict, Any, Optional
import pandas as pd


class BloodDonorDatabase:
    def __init__(self, db_path: str = "data/blood_donor.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        """Ensure all required tables exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS donors (
            donor_id TEXT PRIMARY KEY,
            blood_group TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            city TEXT,
            latitude REAL,
            longitude REAL,
            available INTEGER,
            donation_count INTEGER,
            days_since_last_donation INTEGER,
            previous_requests INTEGER,
            previous_responses INTEGER,
            response_rate REAL,
            contacted_before INTEGER,
            target_response INTEGER
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS emergency_requests (
            request_id TEXT PRIMARY KEY,
            required_blood_group TEXT NOT NULL,
            hospital TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            units_required INTEGER,
            priority TEXT,
            request_date TEXT,
            notes TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS donation_history (
            donation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            donor_id TEXT,
            donation_date TEXT,
            response INTEGER,
            request_id TEXT,
            FOREIGN KEY (donor_id) REFERENCES donors (donor_id)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT,
            donor_id TEXT,
            response_probability REAL,
            recommendation_score REAL,
            recommendation_label TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (donor_id) REFERENCES donors (donor_id)
        )
        """)

        conn.commit()
        conn.close()

    def get_all_donors(self) -> pd.DataFrame:
        """Retrieve all donor records as a Pandas DataFrame."""
        conn = self._get_connection()
        df = pd.read_sql_query("SELECT * FROM donors", conn)
        conn.close()
        return df

    def get_all_requests(self) -> pd.DataFrame:
        """Retrieve all emergency blood requests."""
        conn = self._get_connection()
        df = pd.read_sql_query("SELECT * FROM emergency_requests ORDER BY request_date DESC", conn)
        conn.close()
        return df

    def save_emergency_request(self, request_dict: Dict[str, Any]) -> None:
        """Insert a newly submitted emergency blood request."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO emergency_requests 
        (request_id, required_blood_group, hospital, latitude, longitude, units_required, priority, request_date, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request_dict["request_id"],
            request_dict["required_blood_group"],
            request_dict["hospital"],
            request_dict["latitude"],
            request_dict["longitude"],
            request_dict["units_required"],
            request_dict["priority"],
            request_dict["request_date"],
            request_dict.get("notes", "")
        ))
        conn.commit()
        conn.close()

    def log_predictions(self, predictions_list: List[Dict[str, Any]]) -> None:
        """Log top model recommendation predictions."""
        if not predictions_list:
            return
        conn = self._get_connection()
        cursor = conn.cursor()
        for p in predictions_list:
            cursor.execute("""
            INSERT INTO predictions 
            (request_id, donor_id, response_probability, recommendation_score, recommendation_label)
            VALUES (?, ?, ?, ?, ?)
            """, (
                p["request_id"],
                p["donor_id"],
                p["response_probability"],
                p["recommendation_score"],
                p["recommendation_label"]
            ))
        conn.commit()
        conn.close()

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Fetch high-level KPI dashboard metrics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM donors")
        total_donors = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM donors WHERE available = 1")
        available_donors = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(response_rate) FROM donors")
        avg_response_rate = cursor.fetchone()[0] or 0.0

        cursor.execute("SELECT COUNT(*) FROM emergency_requests")
        total_requests = cursor.fetchone()[0]

        conn.close()
        return {
            "total_donors": total_donors,
            "available_donors": available_donors,
            "avg_response_rate": round(avg_response_rate * 100, 1),
            "total_requests": total_requests
        }
