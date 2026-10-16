import os
import sys
import time
import math
import threading
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Project module imports
from app.DigitalTwinState import DigitalTwinState
from data.DataManager import DataUtil
from generative_ai.AI import GenAI

# Setup paths
ROOT = Path(__file__).resolve().parent
YOLO11_PATH = ROOT / "yolo11n.pt"
WORLD_MODEL_PATH = ROOT / "yolov8s-world.pt"

# Page configuration
st.set_page_config(
    page_title="Smart Waste Digital Twin",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================
# 1. COOL BACKGROUND TEXTURE & FUTURISTIC SPEEDOMETER CSS
# =========================================================================
st.markdown("""
<style>
/* Deep cybernetic grid + constellation dot texture */
.stApp {
    background-color: #080d1a;
    background-image: 
        radial-gradient(circle at 50% 0%, rgba(37, 99, 235, 0.22) 0%, transparent 55%),
        radial-gradient(rgba(56, 189, 248, 0.12) 1.2px, transparent 1.2px),
        linear-gradient(rgba(30, 41, 59, 0.4) 1px, transparent 1px),
        linear-gradient(90deg, rgba(30, 41, 59, 0.4) 1px, transparent 1px);
    background-size: 100% 100%, 28px 28px, 56px 56px, 56px 56px;
    background-position: 0 0, 0 0, 0 0, 0 0;
    color: #f8fafc;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.block-container { padding-top: 0.8rem; max-width: 1480px; }

/* Top Header */
.top-header {
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 16px;
    padding: 16px 22px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}
.header-title { font-size: 26px; font-weight: 800; color: #38bdf8; margin: 0; letter-spacing: -0.5px; }
.header-sub { font-size: 13px; color: #94a3b8; margin-top: 3px; }
.status-pill {
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    background: rgba(34, 197, 94, 0.18);
    color: #4ade80;
    border: 1px solid #22c55e;
}

/* Speedometer Cards */
.speedo-card {
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(51, 65, 85, 0.8);
    border-radius: 18px;
    padding: 12px 8px;
    text-align: center;
    position: relative;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    margin-bottom: 8px;
}
.speedo-card:hover {
    transform: translateY(-3px);
    border-color: #38bdf8;
    box-shadow: 0 6px 22px rgba(56, 189, 248, 0.25);
}
.speedo-card.critical-pulse {
    border-color: #ef4444 !important;
    animation: pulse-red 1.5s infinite;
}
@keyframes pulse-red {
    0% { box-shadow: 0 0 5px rgba(239, 68, 68, 0.4); }
    50% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.8); }
    100% { box-shadow: 0 0 5px rgba(239, 68, 68, 0.4); }
}

.speedo-header { font-size: 13px; font-weight: 700; color: #f1f5f9; margin-bottom: 2px; }
.speedo-meta { font-size: 11px; color: #94a3b8; margin-top: 2px; line-height: 1.4; }

/* Status Badges */
.badge-optimal { background: rgba(34, 197, 94, 0.2); color: #4ade80; padding: 2px 7px; border-radius: 6px; font-size: 10px; font-weight: 700; border: 1px solid #22c55e; }
.badge-filling { background: rgba(234, 179, 8, 0.2); color: #facc15; padding: 2px 7px; border-radius: 6px; font-size: 10px; font-weight: 700; border: 1px solid #eab308; }
.badge-critical { background: rgba(239, 68, 68, 0.2); color: #f87171; padding: 2px 7px; border-radius: 6px; font-size: 10px; font-weight: 700; border: 1px solid #ef4444; }

/* Pop-up Alert Bar */
.popup-alert-bar {
    background: linear-gradient(90deg, rgba(239, 68, 68, 0.25) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
    animation: slideDown 0.3s ease;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
}
@keyframes slideDown {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* KPI Strip */
.kpi-mini {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(51, 65, 85, 0.6);
    border-radius: 12px;
    padding: 10px 12px;
    text-align: center;
}
.kpi-num { font-size: 20px; font-weight: 800; color: #38bdf8; }
.kpi-lbl { font-size: 10.5px; text-transform: uppercase; color: #94a3b8; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# Mouse-pointer bubble/trail animation removed.


# =========================================================================
# 3. FAST CACHED RESOURCE INITIALIZATION
# =========================================================================
@st.cache_resource
def get_twin():
    return DigitalTwinState(cooldown_seconds=1.5, base_dir=str(ROOT))


@st.cache_resource
def get_ai(api_key: str = None):
    return GenAI(api_key=api_key)


twin = get_twin()


# Lazy YOLO loader to keep initial startup < 1.5s
@st.cache_resource
def load_yolo_lazy():
    from ultralytics import YOLO
    if YOLO11_PATH.exists():
        return YOLO(str(YOLO11_PATH)), "YOLO11n"
    elif WORLD_MODEL_PATH.exists():
        return YOLO(str(WORLD_MODEL_PATH)), "YOLO-World"
    return YOLO("yolov8n.pt"), "YOLOv8n"


# =========================================================================
# 4. SPEEDOMETER GAUGE HTML GENERATOR (ZERO INDENTATION TO AVOID CODE PARSING)
# =========================================================================
def render_speedometer_html(bin_obj) -> str:
    """Generates an authentic circular speedometer gauge dial with tick marks and needle."""
    cap = max(0, min(100, bin_obj.capacity))
    color = bin_obj.color
    total_arc = 230.0
    dash_offset = round(total_arc - (cap / 100.0) * total_arc, 1)

    angle_rad = math.radians(-120.0 + (cap * 2.4))
    needle_x = round(70.0 + 40.0 * math.sin(angle_rad), 1)
    needle_y = round(82.0 - 40.0 * math.cos(angle_rad), 1)

    track_color = "#ef4444" if cap >= 75 else ("#eab308" if cap >= 60 else color)
    badge_cls = "badge-critical" if cap >= 75 else ("badge-filling" if cap >= 60 else "badge-optimal")
    card_crit = "critical-pulse" if cap >= 75 else ""
    item_count = len(bin_obj.getItems())

    # Formatted strictly without leading indentation so markdown never parses as code block
    html = (
        f'<div class="speedo-card {card_crit}" style="border-top:3px solid {color};">'
        f'<div class="speedo-header"><span>{bin_obj.icon} {bin_obj.type}</span></div>'
        f'<svg viewBox="0 0 140 110" style="width:100%;max-height:105px;display:block;margin:0 auto;">'
        f'<path d="M 20 82 A 55 55 0 1 1 120 82" fill="none" stroke="#1e293b" stroke-width="9" stroke-linecap="round"/>'
        f'<line x1="28" y1="78" x2="33" y2="73" stroke="#64748b" stroke-width="1.5"/>'
        f'<line x1="70" y1="28" x2="70" y2="34" stroke="#64748b" stroke-width="2"/>'
        f'<line x1="112" y1="78" x2="107" y2="73" stroke="#64748b" stroke-width="1.5"/>'
        f'<path d="M 20 82 A 55 55 0 1 1 120 82" fill="none" stroke="{track_color}" stroke-width="9" stroke-linecap="round" stroke-dasharray="230" stroke-dashoffset="{dash_offset}" style="filter:drop-shadow(0 0 5px {track_color});"/>'
        f'<line x1="70" y1="82" x2="{needle_x}" y2="{needle_y}" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" style="filter:drop-shadow(0 0 2px #fff);"/>'
        f'<circle cx="70" cy="82" r="5" fill="#38bdf8"/>'
        f'<circle cx="70" cy="82" r="2" fill="#0f172a"/>'
        f'<text x="70" y="65" text-anchor="middle" font-size="19" font-weight="800" fill="{track_color}" font-family="sans-serif">{cap}%</text>'
        f'<text x="70" y="102" text-anchor="middle" font-size="9" font-weight="600" fill="#94a3b8" font-family="sans-serif">CAPACITY</text>'
        f'</svg>'
        f'<div style="margin-top:2px;"><span class="{badge_cls}">{bin_obj.status}</span></div>'
        f'<div class="speedo-meta"><b>{item_count}</b> items &bull; <b>{bin_obj.current_weight_kg:.2f} kg</b></div>'
        f'</div>'
    )
    return html


def render_fleet_speedometer_html(overall_pct: int, total_items: int, total_kg: float) -> str:
    """Generates the 7th master gauge showing average fleet fill across all bins."""
    cap = max(0, min(100, overall_pct))
    color = "#38bdf8"  # Always cyan-blue for fleet gauge
    total_arc = 230.0
    dash_offset = round(total_arc - (cap / 100.0) * total_arc, 1)
    angle_rad = math.radians(-120.0 + (cap * 2.4))
    needle_x = round(70.0 + 40.0 * math.sin(angle_rad), 1)
    needle_y = round(82.0 - 40.0 * math.cos(angle_rad), 1)
    track_color = "#ef4444" if cap >= 75 else ("#eab308" if cap >= 60 else "#38bdf8")
    badge_cls = "badge-critical" if cap >= 75 else ("badge-filling" if cap >= 60 else "badge-optimal")
    badge_txt = "CRITICAL" if cap >= 75 else ("FILLING" if cap >= 60 else "OPTIMAL")
    card_crit = "critical-pulse" if cap >= 75 else ""

    html = (
        f'<div class="speedo-card {card_crit}" style="border-top:3px solid #38bdf8;background:rgba(10,20,50,0.95);">'
        f'<div class="speedo-header"><span>🌐 Fleet Average</span></div>'
        f'<svg viewBox="0 0 140 110" style="width:100%;max-height:105px;display:block;margin:0 auto;">'
        f'<path d="M 20 82 A 55 55 0 1 1 120 82" fill="none" stroke="#1e293b" stroke-width="9" stroke-linecap="round"/>'
        f'<line x1="28" y1="78" x2="33" y2="73" stroke="#64748b" stroke-width="1.5"/>'
        f'<line x1="70" y1="28" x2="70" y2="34" stroke="#64748b" stroke-width="2"/>'
        f'<line x1="112" y1="78" x2="107" y2="73" stroke="#64748b" stroke-width="1.5"/>'
        f'<path d="M 20 82 A 55 55 0 1 1 120 82" fill="none" stroke="{track_color}" stroke-width="9" stroke-linecap="round" stroke-dasharray="230" stroke-dashoffset="{dash_offset}" style="filter:drop-shadow(0 0 5px {track_color});"/>'
        f'<line x1="70" y1="82" x2="{needle_x}" y2="{needle_y}" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" style="filter:drop-shadow(0 0 2px #fff);"/>'
        f'<circle cx="70" cy="82" r="5" fill="#38bdf8"/>'
        f'<circle cx="70" cy="82" r="2" fill="#0f172a"/>'
        f'<text x="70" y="65" text-anchor="middle" font-size="19" font-weight="800" fill="{track_color}" font-family="sans-serif">{cap}%</text>'
        f'<text x="70" y="102" text-anchor="middle" font-size="9" font-weight="600" fill="#94a3b8" font-family="sans-serif">FLEET AVG</text>'
        f'</svg>'
        f'<div style="margin-top:2px;"><span class="{badge_cls}">{badge_txt}</span></div>'
        f'<div class="speedo-meta"><b>{total_items}</b> total items &bull; <b>{total_kg:.2f} kg</b></div>'
        f'</div>'
    )
    return html


# =========================================================================
# 5. SIDEBAR
# =========================================================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/recycle-bin.png", width=65)
    st.markdown("### ⚙️ Digital Twin Controls")

    st.markdown("##### 🤖 GenAI Configuration")
    user_api_key = st.text_input("Gemini API Key", type="password", placeholder="Paste AI Studio Key...",
                                 help="Optional: If omitted, built-in intelligent AI analytics engine generates full truthful reports.")
    ai = get_ai(api_key=user_api_key if user_api_key else None)
    if ai.is_live():
        st.success("🟢 Gemini GenAI Live Connected")
    else:
        st.info("⚡ Built-in Heuristic AI Active")

    st.divider()
    st.markdown("##### 👁️ Detection Settings")
    confidence = st.slider("Detection Confidence", 0.15, 0.90, 0.25, 0.05)

    st.divider()
    st.markdown("##### 📍 Campus Zone")
    selected_zone = st.selectbox("Active Monitoring Zone", twin.zones, index=0)
    twin.current_zone = selected_zone

    st.divider()
    st.markdown("##### 🧪 Hackathon Demo Tools")
    if st.button("📊 Load Benchmark Data (7-Day)", use_container_width=True,
                 help="Loads verified sample dataset to demonstrate multi-day telemetry to judges."):
        DataUtil.load_benchmark_demo_data(twin.allbins)
        st.session_state["alert_dismissed"] = False
        st.success("Loaded verified benchmark data!")
        st.rerun()

    if st.button("🔄 Reset All 6 Bins to 0%", use_container_width=True):
        twin.reset()
        st.session_state["alert_dismissed"] = True
        st.session_state.pop("cached_pdf", None)
        st.session_state.pop("cached_ai_data", None)
        st.success("All 6 bins reset to clean 0% state!")
        st.rerun()

# =========================================================================
# 6. TOP HEADER & PROMINENT DOWNLOAD ACTION BAR
# =========================================================================
st.markdown("""
<div class="top-header">
    <div>
        <div class="header-title">♻️ SMART WASTE MANAGEMENT DIGITAL TWIN</div>
        <div class="header-sub">Computer Vision Classification &bull; 6 Speedometer IoT Bins &bull; Generative AI Route & Sustainability Audits</div>
    </div>
    <div>
        <span class="status-pill">🟢 DIGITAL TWIN ACTIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Fetch Truthful Real-Time Metrics
impact = twin.calculate_environmental_impact()
route_opt = twin.optimize_collection_route()
waste_dist = twin.get_waste_category_distribution()

bin_summaries = [
    {
        "Name": b.name,
        "Category": b.type,
        "CurrentFill": b.capacity,
        "TotalItems": len(b.getItems()),
        "EstWeight": b.current_weight_kg,
        "Status": b.status
    }
    for b in twin.allbins
]

# Pop-up Notification for Any Bin Reaching 75% Capacity (With Dismiss Handling)
critical_bins = [b for b in twin.allbins if b.capacity >= 75]
if critical_bins and not st.session_state.get("alert_dismissed", False):
    crit = critical_bins[0]
    p_col1, p_col2 = st.columns([5, 1])
    with p_col1:
        st.markdown(f"""
        <div class="popup-alert-bar">
            <span style="font-size: 26px;">🚨</span>
            <div style="flex: 1;">
                <div style="font-size: 15px; font-weight: 800; color: #f87171;">POP-UP ALERT: {crit.name.upper()} REACHED {crit.capacity}% CAPACITY!</div>
                <div style="font-size: 13px; color: #cbd5e1;">Located in <b>{crit.zone}</b>. Immediate collection dispatch recommended. Logistics routing updated.</div>
            </div>
            <span class="badge-critical" style="font-size: 12px; padding: 4px 10px;">ACTION REQUIRED</span>
        </div>
        """, unsafe_allow_html=True)
    with p_col2:
        if st.button("✖ Dismiss Alert", use_container_width=True):
            st.session_state["alert_dismissed"] = True
            st.rerun()

# TOP ACTION BAR - ONE CLICK BUTTONS AT THE TOP AS REQUESTED
top_col1, top_col2, top_col3, top_col4, top_col5 = st.columns([1.6, 1.2, 1.2, 1.5, 0.8])

with top_col1:
    if st.button("📥 Generate & Download AI Report (PDF)", use_container_width=True, type="primary"):
        with st.spinner("Compiling certified executive report..."):
            from app.report_generator import generate_pdf_report

            ai_data = ai.generate_monthly_report(impact, bin_summaries, waste_dist)
            rep_payload = {
                "title": "Smart Waste Management Digital Twin",
                "month": datetime.now().strftime("%B %Y"),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "bin_summaries": bin_summaries,
                "waste_distribution": waste_dist,
                "environmental_impact": impact,
                "route_optimization": route_opt,
                **ai_data
            }
            st.session_state["cached_pdf"] = generate_pdf_report(rep_payload)
            st.session_state["cached_ai_data"] = ai_data

    if "cached_pdf" in st.session_state:
        st.download_button(
            label="⬇️ Click Here to Save PDF",
            data=st.session_state["cached_pdf"],
            file_name=f"Smart_Waste_Monthly_Report_{datetime.now().strftime('%b_%Y')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

with top_col2:
    if "cached_ai_data" not in st.session_state:
        st.session_state["cached_ai_data"] = ai.generate_monthly_report(impact, bin_summaries, waste_dist)

    from app.report_generator import generate_markdown_report

    rep_md_payload = {
        "title": "Smart Waste Management Digital Twin",
        "month": datetime.now().strftime("%B %Y"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bin_summaries": bin_summaries,
        "waste_distribution": waste_dist,
        "environmental_impact": impact,
        "route_optimization": route_opt,
        **st.session_state["cached_ai_data"]
    }
    md_content = generate_markdown_report(rep_md_payload)
    st.download_button(
        label="📄 Download (Markdown)",
        data=md_content,
        file_name=f"Waste_Audit_{datetime.now().strftime('%b_%Y')}.md",
        mime="text/markdown",
        use_container_width=True
    )

with top_col3:
    history_df = twin.get_usage_timeline_df()
    csv_bytes = history_df.to_csv(index=False) if not history_df.empty else "No records"
    st.download_button(
        label="📊 Export CSV Telemetry",
        data=csv_bytes,
        file_name="waste_telemetry.csv",
        mime="text/csv",
        use_container_width=True
    )

with top_col4:
    demo_choice = st.selectbox(
        "⚡ Quick Test Item:",
        [
            "Plastic Water Bottle", "Cardboard Packaging", "Aluminum Beverage Can",
            "Lithium Smartphone Battery", "Medical Protective Mask", "Banana Peel", "Apple Core"
        ],
        label_visibility="collapsed"
    )

with top_col5:
    if st.button("Deposit", use_container_width=True, help="Simulate waste item deposit directly into Digital Twin"):
        with twin.lock:
            twin.process_detection([demo_choice.lower()], confidence=0.96, zone=twin.current_zone)
        st.session_state["alert_dismissed"] = False
        st.toast(f"Logged {demo_choice} into Digital Twin!", icon="♻️")
        st.rerun()

# =========================================================================
# 7. THE 7 CIRCULAR SPEEDOMETER BINS DISPLAY (6 BINS + 1 FLEET AVERAGE)
# =========================================================================
st.markdown("#### 🧭 Smart Bin Fleet Telemetry (Circular Speedometers)")

# Compute fleet average for the 7th gauge
_fleet_fills = [b.capacity for b in twin.allbins]
_fleet_avg = round(sum(_fleet_fills) / len(_fleet_fills)) if _fleet_fills else 0
_fleet_items = sum(len(b.getItems()) for b in twin.allbins)
_fleet_kg = sum(b.current_weight_kg for b in twin.allbins)

# RENDER ACROSS 7 COLUMNS — 6 individual bins + 1 Fleet Average master gauge
speedo_cols = st.columns(7)
for idx, b in enumerate(twin.allbins):
    with speedo_cols[idx]:
        st.markdown(render_speedometer_html(b), unsafe_allow_html=True)
        if st.button(f"Empty {b.type}", key=f"empty_bin_btn_{b.id}", use_container_width=True):
            b.empty_bin()
            st.session_state["alert_dismissed"] = False
            st.toast(f"Emptied {b.name}!", icon="🚚")
            st.rerun()

# 7th column: Fleet Average Master Gauge (no Empty button — it's a computed summary)
with speedo_cols[6]:
    st.markdown(render_fleet_speedometer_html(_fleet_avg, _fleet_items, _fleet_kg), unsafe_allow_html=True)
    st.caption("All bins combined")

# Executive KPI Strip
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(
        f'<div class="kpi-mini"><div class="kpi-lbl">Total Items Verified</div><div class="kpi-num">{impact["total_items"]}</div></div>',
        unsafe_allow_html=True)
with k2:
    st.markdown(
        f'<div class="kpi-mini"><div class="kpi-lbl">Measured Waste Diverted</div><div class="kpi-num">{impact["diverted_kg"]:.2f} kg</div></div>',
        unsafe_allow_html=True)
with k3:
    st.markdown(
        f'<div class="kpi-mini"><div class="kpi-lbl">Net CO₂ Avoided</div><div class="kpi-num">{impact["co2_saved_kg"]:.2f} kg</div></div>',
        unsafe_allow_html=True)
with k4:
    st.markdown(
        f'<div class="kpi-mini"><div class="kpi-lbl">Recycling Efficiency</div><div class="kpi-num">{impact["recycling_rate"]:.1f}%</div></div>',
        unsafe_allow_html=True)
with k5:
    st.markdown(
        f'<div class="kpi-mini"><div class="kpi-lbl">Collection Fuel Saved</div><div class="kpi-num">{route_opt["fuel_saved_l"]:.2f} L</div></div>',
        unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# =========================================================================
# 8. MAIN DASHBOARD TABS
# =========================================================================
tab_vision, tab_graphs, tab_report, tab_twin, tab_copilot, tab_logs = st.tabs([
    "📷 Live Camera & Waste Detection",
    "📈 Bins Usage Graphs & Analytics",
    "📑 AI Monthly Executive Report",
    "🚚 Fleet Logistics & Route Optimizer",
    "💬 GenAI Twin Copilot",
    "📋 Real Audit & Telemetry Log"
])

# -------------------------------------------------------------------------
# TAB 1: IMAGE SNAPSHOT & DETECTION
# -------------------------------------------------------------------------
with tab_vision:
    st.subheader("👁️ Live Computer Vision Waste Detection")
    st.caption("Capture a webcam photo or upload an image to detect and classify waste items.")

    cv_left, cv_right = st.columns([1.3, 1], gap="large")

    with cv_left:
        cam_source = st.radio(
            "📷 Select Detection Mode:",
            ["📸 Instant Snapshot"],
            horizontal=True,
            help="Capture a webcam photo or upload an image for AI waste detection."
        )

        # ------------------------------------------------------------------
        # MODE 2: INSTANT SNAPSHOT — Capture one frame, run full YOLO on it
        # ------------------------------------------------------------------
        if cam_source == "📸 Instant Snapshot":
            st.caption(
                "📸 **SNAPSHOT** — Capture a photo. AI analyses the full image and instantly adds detected items to bins.")

            snap_col1, snap_col2 = st.columns([1, 1])
            with snap_col1:
                st.markdown("**Option A — Use Webcam**")
                cam_photo = st.camera_input("Take a Photo", label_visibility="collapsed")
            with snap_col2:
                st.markdown("**Option B — Upload Image**")
                uploaded_snap = st.file_uploader(
                    "Upload a waste photo", type=["jpg", "jpeg", "png"],
                    label_visibility="collapsed"
                )

            # Pick whichever source has data (webcam takes priority)
            snap_bytes = cam_photo.getvalue() if cam_photo else (uploaded_snap.getvalue() if uploaded_snap else None)

            if snap_bytes is not None:
                with st.spinner("🔍 Running AI detection on snapshot..."):
                    model, model_tag = load_yolo_lazy()
                    img_buf = np.frombuffer(snap_bytes, dtype=np.uint8)
                    snap_frame = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)

                    if snap_frame is not None:
                        # Full-resolution YOLO on snapshot (no frame-skip needed — single shot)
                        results = model(snap_frame, conf=confidence, imgsz=640, verbose=False)
                        annotated_snap = results[0].plot()
                        labels_snap = []
                        if results[0].boxes is not None:
                            for c in results[0].boxes.cls:
                                labels_snap.append(model.names[int(c)].lower())

                        # Show the annotated snapshot
                        st.image(
                            cv2.cvtColor(annotated_snap, cv2.COLOR_BGR2RGB),
                            use_container_width=True,
                            caption=f"🤖 {model_tag} Snapshot Detection | imgsz=640 | conf≥{confidence}"
                        )

                        if labels_snap:
                            with twin.lock:
                                updated, added = twin.process_detection(
                                    labels_snap, confidence=confidence, zone=twin.current_zone
                                )
                            if updated:
                                items_str = ", ".join(f"**{a['Item']}** → **{a['Bin']}**" for a in added)
                                st.success(f"✅ Detected & logged: {items_str}")
                                st.balloons()
                                hit_crit = [b for b in twin.allbins if b.capacity >= 75]
                                if hit_crit:
                                    st.session_state["alert_dismissed"] = False
                                    st.toast(f"🚨 {hit_crit[0].name} at {hit_crit[0].capacity}%!", icon="⚠️")
                            else:
                                st.info("Items already recently logged — cooldown active.")
                        else:
                            st.warning(
                                "⚠️ No waste items detected in this image. Try a clearer photo or lower the confidence slider in the sidebar.")
                    else:
                        st.error("Could not decode image. Please try a different photo.")

        # ------------------------------------------------------------------
    # RIGHT PANEL: Latest detection + segregation guide
    # ------------------------------------------------------------------
    with cv_right:
        st.markdown("##### 🎯 Latest Detection Result")
        with twin.lock:
            recent_events = list(twin.events[:4])

        if recent_events:
            last_ev = recent_events[0]
            st.markdown(
                f'<div style="padding:14px;border-radius:12px;background:rgba(15,23,42,0.9);'
                f'border:2px solid {last_ev["Color"]};margin-bottom:14px;'
                f'box-shadow:0 0 12px {last_ev["Color"]}30;">'
                f'<span style="font-size:11px;color:#94a3b8;text-transform:uppercase;">Last Classified Item</span>'
                f'<div style="font-size:22px;font-weight:700;color:#f8fafc;margin:6px 0;">'
                f'{last_ev["Icon"]} {last_ev["Item"]}</div>'
                f'<div style="font-size:13px;color:{last_ev["Color"]};font-weight:600;">'
                f'Logged to: {last_ev["Bin"]} — {last_ev["Fill"]}</div>'
                f'<div style="font-size:11px;color:#94a3b8;margin-top:4px;">'
                f'Conf: {last_ev["Confidence"]} &bull; {last_ev["Time"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown("##### 📊 Recent Detection Log")
            for ev in recent_events:
                st.markdown(
                    f'<div style="padding:6px 10px;border-radius:8px;background:rgba(15,23,42,0.7);'
                    f'border-left:3px solid {ev["Color"]};margin-bottom:5px;font-size:12px;">'
                    f'{ev["Icon"]} <b>{ev["Item"]}</b> → {ev["Bin"]} ({ev["Fill"]}) '
                    f'<span style="color:#64748b;font-size:10px;">{ev["Time"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No detections yet. Start the camera or take a snapshot.")

        st.divider()
        st.markdown("##### 💡 Waste Segregation Guide")
        st.markdown("""
- 🔵 **Plastic** — Bottles, cups, bags, wrappers
- 📦 **Paper** — Cardboard, documents, dry napkins
- 🥫 **Metal** — Cans, tins, aluminium foil
- ⚡ **Electronics** — Batteries, chargers, cables, phones
- ☣️ **Medical** — Masks, gloves, blister packs
- 🍏 **Organic** — Peels, food scraps, leaves
        """)

# -------------------------------------------------------------------------
# TAB 2: BINS USAGE GRAPHS & ANALYTICS
# -------------------------------------------------------------------------
with tab_graphs:
    st.subheader("📊 Interactive Bin Usage Analytics")
    st.caption("Track exact capacity utilization, item counts, and category distribution across all 6 smart bins.")

    comp_df = twin.get_bin_fill_comparison_df()

    g_col1, g_col2 = st.columns([1.2, 1], gap="large")

    with g_col1:
        st.markdown("##### 📏 Current Bin Fill Percentage (Speedometer Values)")
        fill_chart_df = comp_df.set_index("Type")[["Fill %"]]
        st.bar_chart(fill_chart_df, color="#38bdf8", use_container_width=True)

    with g_col2:
        st.markdown("##### 🔢 Total Items Users Disposed In Each Bin")
        items_chart_df = comp_df.set_index("Type")[["Items Logged"]]
        st.bar_chart(items_chart_df, color="#22c55e", use_container_width=True)

    st.divider()

    g_col3, g_col4 = st.columns([1, 1], gap="large")

    with g_col3:
        st.markdown("##### 🥧 Usage Share by Waste Category")
        active_counts = {k: v for k, v in waste_dist.items() if v > 0}
        if active_counts:
            dist_df = pd.DataFrame(list(active_counts.items()), columns=["Category", "Count"])
            st.bar_chart(dist_df.set_index("Category"), color="#8b5cf6", use_container_width=True)
        else:
            st.info(
                "No items deposited yet. Present waste to the camera or use the Deposit button above to generate usage charts.")

    with g_col4:
        st.markdown("##### ⏱️ User Disposal Activity Timeline")
        hist_df = twin.get_usage_timeline_df()
        if not hist_df.empty:
            timeline_summary = hist_df.copy()
            timeline_summary["Hour"] = timeline_summary["Timestamp"].dt.strftime("%H:%M")
            hourly_counts = timeline_summary.groupby(["Hour", "Category"]).size().unstack(fill_value=0)
            st.area_chart(hourly_counts, use_container_width=True)
        else:
            st.caption("Disposal timeline will plot automatically as waste is detected.")

# -------------------------------------------------------------------------
# TAB 3: AI MONTHLY EXECUTIVE REPORT
# -------------------------------------------------------------------------
with tab_report:
    st.subheader("📑 Certified Monthly Sustainability Audit")
    st.caption(f"Period: {datetime.now().strftime('%B %Y')} &bull; Grounded in 100% Genuine Telemetry")

    rep_c1, rep_c2 = st.columns([1.5, 1], gap="large")

    ai_data = st.session_state.get("cached_ai_data", ai.generate_monthly_report(impact, bin_summaries, waste_dist))

    with rep_c1:
        st.markdown("### 1. Executive Summary")
        st.info(ai_data.get("executive_summary"))

        st.markdown("### 2. Generative AI Strategic Recommendations")
        recs = ai_data.get("recommendations", [])
        for idx, r in enumerate(recs, 1):
            st.markdown(f"**{idx}.** {r}")

        st.markdown("### 3. Smart Logistics Dispatch Plan")
        st.write(ai_data.get("route_plan"))

    with rep_c2:
        st.markdown("### 4. Verified Monthly KPIs")
        st.markdown(f"""
        <div class="speedo-card" style="text-align: left; padding: 16px;">
            <b>Total Verified Items:</b> {impact['total_items']}<br>
            <b>Measured Waste Diverted:</b> {impact['diverted_kg']:.2f} kg<br>
            <b>Net CO₂ Saved:</b> {impact['co2_saved_kg']:.2f} kg<br>
            <b>Recycling Rate:</b> {impact['recycling_rate']:.1f}%<br>
            <b>Collection Fuel Saved:</b> {route_opt['fuel_saved_l']:.2f} Liters
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 5. Download Certified Report")
        if "cached_pdf" in st.session_state:
            st.download_button(
                label="📥 Download Certified PDF",
                data=st.session_state["cached_pdf"],
                file_name=f"Smart_Waste_Report_{datetime.now().strftime('%b_%Y')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
                key="rep_dl_pdf_tab"
            )
        else:
            if st.button("Compile PDF Now", use_container_width=True, type="primary"):
                from app.report_generator import generate_pdf_report

                rep_payload = {
                    "title": "Smart Waste Management Digital Twin",
                    "month": datetime.now().strftime("%B %Y"),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "bin_summaries": bin_summaries,
                    "waste_distribution": waste_dist,
                    "environmental_impact": impact,
                    "route_optimization": route_opt,
                    **ai_data
                }
                st.session_state["cached_pdf"] = generate_pdf_report(rep_payload)
                st.rerun()

# -------------------------------------------------------------------------
# TAB 4: FLEET LOGISTICS & ROUTE OPTIMIZER
# -------------------------------------------------------------------------
with tab_twin:
    st.subheader("🚚 Collection Logistics & Dynamic Route Optimizer")
    st.caption("Automated vehicle dispatch based on real-time fill thresholds (75%).")

    r_col1, r_col2 = st.columns([1.2, 1], gap="large")

    with r_col1:
        if route_opt["dispatched"]:
            st.error(route_opt["message"])
            st.markdown("##### 📍 Priority Collection Route:")
            for step_i, step_node in enumerate(route_opt["route"], 1):
                st.markdown(f"**Stop {step_i}:** {step_node}")
        else:
            st.success(route_opt["message"])
            st.markdown("##### 📍 Route Status:")
            st.markdown("&bull; Collection vehicles on standby at Central Depot.")

    with r_col2:
        st.markdown("##### 🌿 Logistics Efficiency Gains")
        m1, m2 = st.columns(2)
        m1.metric("Distance Saved", f"{route_opt['distance_saved_km']:.1f} km")
        m2.metric("Fuel Saved", f"{route_opt['fuel_saved_l']:.2f} L")
        st.metric("Net CO₂ Avoided", f"{route_opt['co2_avoided_kg']:.2f} kg")

# -------------------------------------------------------------------------
# TAB 5: GENAI TWIN COPILOT (ANSWERS USER'S EXACT QUERY)
# -------------------------------------------------------------------------
with tab_copilot:
    st.subheader("💬 Ask WasteTwin AI Copilot")
    st.caption("Directly answers questions about waste segregation, fill levels, e-waste, or collection routes.")

    copilot_query = st.chat_input(
        "Ask a question (e.g. 'Can I recycle batteries?', 'Where does a pizza box go?', or 'Which bin is fullest?')...")

    if "copilot_history" not in st.session_state:
        st.session_state.copilot_history = [
            ("assistant",
             "Hello! I am your Smart Waste Management Digital Twin Copilot. Ask me where any waste item belongs, check real-time bin levels, or inquire about collection logistics.")
        ]

    if copilot_query:
        st.session_state.copilot_history.append(("user", copilot_query))
        twin_context = {
            "bins": [{"name": b.name, "capacity": b.capacity, "zone": b.zone} for b in twin.allbins],
            "bins_summary": [f"{b.name}: {b.capacity}%" for b in twin.allbins],
            "total_items": impact["total_items"],
            "diverted_kg": impact["diverted_kg"],
            "co2_saved_kg": impact["co2_saved_kg"],
            "zone": twin.current_zone
        }
        answer = ai.ask_copilot(copilot_query, twin_context)
        st.session_state.copilot_history.append(("assistant", answer))

    for role, text in st.session_state.copilot_history:
        with st.chat_message(role):
            st.markdown(text)

# -------------------------------------------------------------------------
# TAB 6: REAL AUDIT & TELEMETRY LOG
# -------------------------------------------------------------------------
with tab_logs:
    st.subheader("📋 Verified Audit Trail & Telemetry Records")

    l_c1, l_c2 = st.columns(2, gap="medium")

    with l_c1:
        st.markdown("##### ⚡ Recent Detection Stream")
        with twin.lock:
            audit_events = list(twin.events)
        if audit_events:
            st.dataframe(pd.DataFrame(audit_events), use_container_width=True, hide_index=True)
        else:
            st.caption("No detections logged yet.")

    with l_c2:
        st.markdown("##### 🗄️ Stored CSV Records")
        hist = twin.get_usage_timeline_df()
        if not hist.empty:
            st.dataframe(hist.tail(100), use_container_width=True, hide_index=True)
        else:
            st.caption("CSV files are currently empty.")

