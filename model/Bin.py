import os
import time
from data.DataManager import DataUtil

class Bin:
    """
    Virtual Smart Bin Digital Twin Model.
    Represents an individual physical bin node within the IoT waste network.
    """
    
    ITEM_WEIGHT_ESTIMATES = {
        "Plastic": 0.08,      # 80g avg
        "Paper": 0.12,        # 120g avg
        "Metal": 0.18,        # 180g avg
        "Electronics": 0.35,  # 350g avg
        "Medical": 0.06,      # 60g avg
        "Organic": 0.28       # 280g avg
    }

    def __init__(self, id: int, name: str, capacity: int, type: str, path: str, color: str, icon: str, zone: str = "Main Hub"):
        self.id = id
        self.name = name
        self.capacity = capacity
        self.type = type
        self.path = path
        self.color = color
        self.icon = icon
        self.zone = zone
        self.max_capacity = 100
        self.max_weight_kg = 35.0
        self.temperature_c = 22.0
        self.last_updated = "Never"
        self.items = DataUtil()
        
        # Load genuine saved data if present
        if os.path.exists(path) and os.path.getsize(path) > 0:
            self.items.readFromCSV(path)
            if not self.items.df.empty and "Capacity" in self.items.df.columns:
                last_cap = self.items.df["Capacity"].dropna()
                if not last_cap.empty:
                    self.capacity = int(last_cap.iloc[-1])
                if "Timestamp" in self.items.df.columns and not self.items.df["Timestamp"].empty:
                    self.last_updated = str(self.items.df["Timestamp"].iloc[-1])

    def __str__(self):
        return f"{self.name} ({self.type}) - Fill: {self.capacity}%"

    @property
    def status(self) -> str:
        if self.capacity >= 85:
            return "Critical"
        elif self.capacity >= 60:
            return "Filling"
        return "Optimal"

    @property
    def current_weight_kg(self) -> float:
        unit_wt = self.ITEM_WEIGHT_ESTIMATES.get(self.type, 0.15)
        count = len(self.getItems())
        return round(min(self.max_weight_kg, count * unit_wt), 2)

    def addItem(self, item: dict):
        """Logs a genuine item deposit and increases fill level."""
        new_cap = min(100, item.get("Capacity", self.capacity + 8))
        self.capacity = new_cap
        self.last_updated = item.get("Timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        item["Capacity"] = new_cap
        self.items.addRow(item, path=self.path)

    def empty_bin(self):
        """Simulates municipal collection emptying this bin."""
        self.capacity = 0
        self.temperature_c = 21.5
        self.last_updated = f"{time.strftime('%H:%M:%S')} (Emptied)"
        self.items.createDataframe([])
        if self.path and os.path.exists(self.path):
            try:
                open(self.path, "w").close()
            except Exception:
                pass

    def getItems(self) -> list:
        return self.items.getRows()

    def getDataFrame(self):
        return self.items.parseDataframe()

    def reset(self):
        """Completely purges bin state and deletes CSV log."""
        self.capacity = 0
        self.last_updated = "Reset"
        self.items.createDataframe([])
        if self.path and os.path.exists(self.path):
            try:
                open(self.path, "w").close()
            except Exception:
                pass
