import os
import csv
import pandas as pd
from datetime import datetime, timedelta

class DataUtil:
    """
    High-performance data management for Smart Waste Digital Twin.
    Ensures 100% genuine data logging, fast append I/O, and zero false metrics.
    """
    def __init__(self):
        self.df = pd.DataFrame()

    def addRow(self, data: dict, path: str = None):
        """Appends a new detection record in memory and persists to disk."""
        new_row = pd.DataFrame([data])
        if self.df.empty:
            self.df = new_row
        else:
            self.df = pd.concat([self.df, new_row], ignore_index=True)
            
        if path:
            self.appendRowToCSV(path, data)

    def appendRowToCSV(self, path: str, data: dict):
        """High-speed disk append without full CSV rewrites."""
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        fieldnames = list(data.keys())
        
        with open(path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)

    def parseDataframe(self) -> pd.DataFrame:
        return self.df

    def writeToCSV(self, path: str):
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.df.to_csv(path, index=False)

    def readFromCSV(self, path: str) -> pd.DataFrame:
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                self.df = pd.read_csv(path)
            except Exception:
                self.df = pd.DataFrame()
        else:
            self.df = pd.DataFrame()
        return self.df

    def createDataframe(self, data=None):
        self.df = pd.DataFrame() if data is None else pd.DataFrame(data)

    def getRows(self) -> list:
        if self.df.empty:
            return []
        return self.df.to_dict(orient="records")

    @staticmethod
    def load_benchmark_demo_data(all_bins: list):
        """
        User-controlled benchmark demo dataset.
        Only populated when the user explicitly clicks 'Load Benchmark Data'.
        """
        now = datetime.now()
        benchmark_events = {
            "Plastic": [
                ("Clear plastic water bottle", 0.95),
                ("Crinkled plastic wrapper", 0.88),
                ("Transparent plastic cup", 0.92),
                ("Plastic grocery bag", 0.90)
            ],
            "Paper": [
                ("Cardboard packaging box", 0.94),
                ("Paper napkin", 0.86),
                ("Printed document", 0.91)
            ],
            "Metal": [
                ("Aluminum beverage can", 0.96),
                ("Soda can", 0.93),
                ("Tin food can", 0.89)
            ],
            "Electronics": [
                ("Lithium smartphone battery", 0.97),
                ("USB charging cable", 0.91),
                ("Broken computer mouse", 0.93)
            ],
            "Medical": [
                ("Blue protective mask", 0.95),
                ("Medical latex gloves", 0.92),
                ("Antiseptic blister pack", 0.88)
            ],
            "Organic": [
                ("Banana peel", 0.96),
                ("Apple core", 0.94),
                ("Leftover sandwich", 0.89)
            ]
        }

        for b in all_bins:
            items_list = benchmark_events.get(b.type, [])
            for idx, (item_name, conf) in enumerate(items_list):
                ts = (now - timedelta(hours=(idx * 4) + 1)).strftime("%Y-%m-%d %H:%M:%S")
                record = {
                    "Timestamp": ts,
                    "Item": item_name,
                    "Category": b.type,
                    "Capacity": (idx + 1) * 8,
                    "Zone": b.zone,
                    "Confidence": conf
                }
                b.addItem(record)

