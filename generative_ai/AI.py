import os
import json
import re
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class GenAI:
    """
    Intelligent Generative AI Engine for Smart Waste Digital Twin.
    Provides direct, accurate answers to user queries, certified sustainability reports,
    and smart route logistics intelligence.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        self.model_name = "gemini-2.5-flash"

        if HAS_GENAI and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    def is_live(self) -> bool:
        return self.client is not None and bool(self.api_key)

    def execute_prompt(self, prompt: str) -> str:
        if not self.is_live():
            return ""
        for m in [self.model_name, "gemini-2.0-flash", "gemini-1.5-flash"]:
            try:
                response = self.client.models.generate_content(
                    model=m,
                    contents=prompt
                )
                if response and hasattr(response, "text") and response.text:
                    return response.text.strip()
            except Exception:
                continue
        return ""

    def generate_monthly_report(self, summary_stats: dict, bin_summaries: list, waste_distribution: dict) -> dict:
        total_items = summary_stats.get("total_items", 0)
        diverted_kg = summary_stats.get("diverted_kg", 0.0)
        co2_saved_kg = summary_stats.get("co2_saved_kg", 0.0)
        recycling_rate = summary_stats.get("recycling_rate", 0.0)

        if total_items == 0:
            return {
                "executive_summary": (
                    "Smart Waste Management Digital Twin has initialized successfully. "
                    "All 6 intelligent waste bins (Plastic, Paper, Metal, Electronics, Medical, Organic) "
                    "are currently online at baseline (0% fill level). No waste deposits have been recorded yet. "
                    "Start live camera detection or deposit an item to begin generating real-time analytics."
                ),
                "recommendations": [
                    "Perform initial camera calibration across all 6 sorting bins.",
                    "Verify visual signage for Plastic, Paper, Metal, Electronics, Medical, and Organic streams.",
                    "Test live object recognition with everyday items (bottles, paper, cans, batteries).",
                    "Maintain the automated route dispatch trigger at 75% capacity to optimize collection."
                ],
                "route_plan": "Collection vehicles are currently on standby. Zero bin overflow risks detected."
            }

        if self.is_live():
            prompt = f"""
You are the AI Chief Sustainability Officer for a Smart Waste Management Digital Twin.
Analyze this ACTUAL real-time waste dataset and return a JSON object with:
1. "executive_summary": (3 sentences analyzing the TRUE recorded numbers)
2. "recommendations": (4 specific, actionable recommendations tailored directly to the waste types detected)
3. "route_plan": (A 2-sentence logistics instruction for municipal collection trucks)

REAL TELEMETRY:
- Total Items Logged: {total_items}
- Measured Mass Diverted: {diverted_kg:.2f} kg
- Estimated CO2 Avoided: {co2_saved_kg:.2f} kg
- Measured Recycling Rate: {recycling_rate:.1f}%
- Bins Status: {bin_summaries}
- Waste Distribution: {waste_distribution}

Output JSON strictly:
{{"executive_summary": "...", "recommendations": ["...", "...", "...", "..."], "route_plan": "..."}}
"""
            try:
                res = self.execute_prompt(prompt)
                clean_res = res.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean_res)
                if "executive_summary" in parsed and "recommendations" in parsed:
                    return parsed
            except Exception:
                pass

        active_counts = {k: v for k, v in waste_distribution.items() if v > 0}
        top_cat = max(active_counts, key=active_counts.get) if active_counts else "General Waste"
        highest_bin = max(bin_summaries, key=lambda b: b.get("CurrentFill", 0)) if bin_summaries else {"Name": "Plastic Bin", "CurrentFill": 0}

        exec_summary = (
            f"During this operating cycle, the Digital Twin monitored and classified {total_items} genuine waste items, "
            f"diverting {diverted_kg:.2f} kg of sorted materials and avoiding an estimated {co2_saved_kg:.2f} kg of CO₂ emissions. "
            f"The primary waste volume was concentrated in {top_cat} ({waste_distribution.get(top_cat, 0)} items), "
            f"with {highest_bin.get('Name')} reaching the highest utilization at {highest_bin.get('CurrentFill')}% capacity."
        )

        recs = [
            f"Increase collection cadence for {highest_bin.get('Name')} to prevent overflow as it approaches capacity limits.",
            f"Reinforce educational signage for {top_cat} disposal to maintain high segregation purity.",
            "Promote e-waste and battery drop-offs in dedicated Tech Lab collection kiosks to recover precious materials.",
            "Utilize the Digital Twin's dynamic route dispatch to dispatch collection vehicles only to bins exceeding 75% fill level."
        ]

        route_plan = (
            f"Logistics dispatch prioritized high-volume stations including {highest_bin.get('Name')} ({highest_bin.get('CurrentFill')}% fill). "
            "Point-to-point collection eliminated unnecessary stops, saving vehicle fuel and reducing municipal carbon emissions."
        )

        return {
            "executive_summary": exec_summary,
            "recommendations": recs,
            "route_plan": route_plan
        }

    def ask_copilot(self, question: str, twin_context: dict) -> str:
        """
        Directly answers the user's exact query with high accuracy and domain relevance.
        """
        user_query = question.strip()
        q_lower = user_query.lower()

        # 1. If Gemini is live, let Gemini answer with strict instruction to be direct
        if self.is_live():
            prompt = f"""
You are the WasteTwin Copilot AI for a Smart Waste Management Digital Twin.
Answer the user's question directly, accurately, and helpfuly.
Do NOT give generic introductions or repetitive statements. Address their exact question.

Current Digital Twin State:
- Bins Status: {twin_context.get('bins', [])}
- Total Items Logged: {twin_context.get('total_items', 0)}
- Measured Diverted Mass: {twin_context.get('diverted_kg', 0)} kg
- Net CO2 Saved: {twin_context.get('co2_saved_kg', 0)} kg
- Current Zone: {twin_context.get('zone', 'Main Campus')}

User Question: "{user_query}"

Direct, specific answer (2-4 sentences):
"""
            res = self.execute_prompt(prompt)
            if res:
                return res

        # 2. Comprehensive Semantic Fallback Engine (Answers ANY user query directly)
        bins = twin_context.get("bins", [])
        total_items = twin_context.get("total_items", 0)
        diverted_kg = twin_context.get("diverted_kg", 0.0)
        co2_saved = twin_context.get("co2_saved_kg", 0.0)

        # GREETINGS
        if re.search(r"\b(hi|hello|hey|greetings|morning|afternoon|evening)\b", q_lower):
            return (
                "Hello! I am your Smart Waste Management Digital Twin Copilot. "
                "You can ask me where any specific waste item should be thrown, check which bins are full, "
                "inquire about collection routes, or learn how to reduce contamination."
            )

        # WHO ARE YOU / CAPABILITIES
        if "who are you" in q_lower or "what can you do" in q_lower or "help" in q_lower:
            return (
                "I am the GenAI Copilot integrated into this Smart Waste Digital Twin. "
                "I monitor real-time fill telemetry across the 6 smart bins (Plastic, Paper, Metal, Electronics, Medical, Organic), "
                "answer segregation questions, calculate carbon footprint savings, and guide collection truck route optimization."
            )

        # SPECIFIC ITEM SEGREGATION QUESTIONS
        if any(w in q_lower for w in ["battery", "batteries", "cell", "lithium", "aa battery"]):
            return (
                "Batteries (AA, AAA, button cells, lithium-ion) must ALWAYS be deposited in the ⚡ **Electronics / E-Waste Bin**. "
                "Never mix batteries into plastic or general trash, as mechanical compaction in garbage trucks can rupture cells and spark dangerous fires."
            )

        if any(w in q_lower for w in ["plastic bottle", "water bottle", "soda bottle"]):
            return (
                "Plastic bottles go into the 🔵 **Plastic Bin**. "
                "For optimal recycling, please empty all remaining liquid, replace the cap, and slightly crush the bottle to conserve bin volume."
            )

        if any(w in q_lower for w in ["wrapper", "chip bag", "packet", "chips", "biscuit", "crisps"]):
            return (
                "Plastic snack wrappers and crinkled packaging bags belong in the 🔵 **Plastic Bin**. "
                "Ensure they are free of excess food particles so they can be processed in municipal polymer recovery lines."
            )

        if any(w in q_lower for w in ["cup", "plastic cup"]):
            return (
                "Transparent plastic cups belong in the 🔵 **Plastic Bin**. "
                "If it is a wax-coated paper coffee cup, check if the lid is plastic (goes to Plastic) while the cup body goes to Paper if clean."
            )

        if any(w in q_lower for w in ["pizza box", "cardboard box", "carton", "packaging box"]):
            return (
                "Clean cardboard boxes belong in the 📦 **Paper & Cardboard Bin**. "
                "For pizza boxes, if the lid is clean cardboard it can be recycled; if the base is heavily soaked with grease and oil, tear off the greasy portion and place it in Organic waste."
            )

        if any(w in q_lower for w in ["napkin", "tissue", "paper towel", "toilet paper"]):
            return (
                "Dry, clean paper napkins and paper scraps belong in the 📦 **Paper Bin**. "
                "However, if a napkin is heavily soiled with food oils or grease, it should go into the 🍏 **Organic Bin** for composting."
            )

        if any(w in q_lower for w in ["can", "soda can", "aluminum", "tin", "beverage can", "coke can"]):
            return (
                "Aluminum beverage cans and tin food cans belong in the 🥫 **Metal Bin**. "
                "Aluminum is 100% infinitely recyclable and requires 95% less energy to recycle than smelting new bauxite ore!"
            )

        if any(w in q_lower for w in ["phone", "smartphone", "laptop", "cable", "charger", "wire", "mouse", "keyboard", "circuit"]):
            return (
                "Broken smartphones, cables, laptop chargers, keyboards, and circuit boards belong strictly in the ⚡ **Electronics / E-Waste Bin**. "
                "This enables the recovery of valuable gold, copper, and rare-earth metals while preventing toxic lead and cadmium leaching."
            )

        if any(w in q_lower for w in ["mask", "syringe", "needle", "glove", "latex", "medicine", "pill", "blister", "bandage"]):
            return (
                "Medical masks, latex gloves, syringes, bandages, and medicine blister packs must be deposited into the ☣️ **Medical & Hazardous Bin**. "
                "This isolates bio-hazardous and sharp materials for regulated high-temperature sanitary incineration."
            )

        if any(w in q_lower for w in ["banana", "apple", "peel", "food", "fruit", "leftover", "vegetable", "bread", "sandwich"]):
            return (
                "Fruit peels, apple cores, food scraps, and vegetable leftovers belong in the 🍏 **Organic / Food Waste Bin**. "
                "These biodegradable materials are diverted from landfills to commercial composting facilities and anaerobic biogas digesters."
            )

        if any(w in q_lower for w in ["styrofoam", "thermocol"]):
            return (
                "Styrofoam (expanded polystyrene) is generally not accepted in standard plastic recycling due to its low density and high contamination rate. "
                "Unless your municipality has a specialized EPS densification drop-off, it is treated as general residual waste."
            )

        if any(w in q_lower for w in ["glass", "broken glass", "bulb", "light bulb"]):
            return (
                "Clean glass bottles belong in glass recycling stations. Broken glass and incandescent/CFL bulbs should be carefully wrapped and handled as hazardous/residual waste to protect sanitation personnel."
            )

        # STATUS AND TELEMETRY QUERIES
        if any(w in q_lower for w in ["fullest", "highest", "most full", "which bin is full"]):
            if not bins or all(b.get("capacity", 0) == 0 for b in bins):
                return "Currently, all 6 bins are at 0% fill level. None of the bins are full."
            fullest = max(bins, key=lambda b: b.get("capacity", 0))
            return f"The fullest bin right now is the **{fullest.get('name')}** at **{fullest.get('capacity')}%** capacity located in {fullest.get('zone')}."

        if any(w in q_lower for w in ["emptiest", "lowest"]):
            if not bins:
                return "All bins are currently at 0% capacity."
            emptiest = min(bins, key=lambda b: b.get("capacity", 0))
            return f"The bin with the lowest fill is the **{emptiest.get('name')}** at **{emptiest.get('capacity')}%** capacity."

        if any(w in q_lower for w in ["how many items", "total items", "item count"]):
            return f"A total of **{total_items} items** have been detected and segregated across all 6 smart bins in the current session."

        if any(w in q_lower for w in ["co2", "carbon", "emissions", "saved"]):
            return f"By segregating waste at the source, the system has diverted **{diverted_kg:.2f} kg** of waste, saving an estimated **{co2_saved:.2f} kg of CO₂ equivalent**."

        if any(w in q_lower for w in ["truck", "route", "collection", "dispatch", "pickup"]):
            critical = [b for b in bins if b.get("capacity", 0) >= 75]
            if critical:
                names = ", ".join([b.get("name") for b in critical])
                return f"Collection truck dispatch is active! The following bin(s) exceeded the 75% threshold: **{names}**. The dynamic routing algorithm has computed the shortest collection path."
            else:
                return "All bins are currently below the 75% critical threshold. The collection truck remains on standby at the depot, conserving fuel and reducing emissions."

        if any(w in q_lower for w in ["digital twin", "what is digital twin", "how it works", "ecosystem"]):
            return (
                "A Digital Twin is a real-time virtual replica of physical assets. In our system, Computer Vision identifies waste items, "
                "the Digital Twin mirrors the exact fill level, weight, and status of physical bins across campus zones, "
                "and Generative AI analyzes trends to dispatch collection trucks and produce certified sustainability audits."
            )

        if any(w in q_lower for w in ["contamination", "wrong bin", "cleanliness", "segregation"]):
            return (
                "Waste contamination occurs when items are deposited into the wrong stream (e.g. greasy food in paper bins or batteries in plastic). "
                "Our Computer Vision model prevents this by performing real-time classification and directing users with colored bounding boxes before the item is thrown."
            )

        # GENERAL CONTEXTUAL FALLBACK (Directly references the user's words)
        return (
            f"Regarding '{user_query}': The Smart Waste Digital Twin manages 6 distinct streams (Plastic, Paper, Metal, Electronics, Medical, Organic). "
            f"Currently {total_items} items are recorded across the fleet with {diverted_kg:.2f} kg diverted. "
            "For specific disposal guidance, specify the item name (e.g. 'can I recycle batteries?' or 'where does a soda can go?')."
        )