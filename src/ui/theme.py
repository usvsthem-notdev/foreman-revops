"""
Load-Bearing Light design system.
Bone paper · slate ink · sage (local) · clay (frontier).
"""

# Brand palette
BONE    = "#F2EDE0"
SLATE   = "#1C2635"
SLATE_2 = "#2D3748"
SAGE    = "#6B9E78"     # absorbed locally
CLAY    = "#C4714A"     # frontier spend
SAND    = "#D4C5A9"
MUTED   = "#8A9BB0"

# Plotly-ready color sequences
PLOTLY_COLORS = [SAGE, CLAY, "#A8C5B5", "#D4946A", SAND, MUTED]

CSS = f"""
<style>
/* ---- global ---- */
html, body, [class*="css"] {{
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    background-color: {BONE};
    color: {SLATE};
}}

/* ---- sidebar ---- */
section[data-testid="stSidebar"] {{
    background-color: {SLATE};
}}
section[data-testid="stSidebar"] * {{
    color: {BONE} !important;
}}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stDateInput label {{
    color: {SAND} !important;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}

/* ---- metric cards ---- */
[data-testid="stMetric"] {{
    background: white;
    border: 1px solid {SAND};
    border-radius: 2px;
    padding: 1rem 1.25rem;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {MUTED};
}}
[data-testid="stMetricValue"] {{
    font-size: 1.6rem;
    font-weight: 600;
    color: {SLATE};
}}

/* ---- tabs ---- */
button[data-baseweb="tab"] {{
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {MUTED};
    border-bottom: 2px solid transparent;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {SLATE};
    border-bottom: 2px solid {CLAY};
    font-weight: 600;
}}

/* ---- section headers ---- */
.foreman-section {{
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {MUTED};
    border-bottom: 1px solid {SAND};
    padding-bottom: 0.25rem;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}}

/* ---- finding cards ---- */
.finding-high {{
    border-left:3px solid {CLAY}; background:#FDF4F0;
    padding:.75rem 1rem; margin:.5rem 0; border-radius:2px;
}}
.finding-medium {{
    border-left:3px solid #E8B86D; background:#FDFAF0;
    padding:.75rem 1rem; margin:.5rem 0; border-radius:2px;
}}
.finding-low {{
    border-left:3px solid {SAGE}; background:#F0FAF3;
    padding:.75rem 1rem; margin:.5rem 0; border-radius:2px;
}}

/* ---- pill badges ---- */
.pill-sage {{ background: {SAGE}22; color: {SAGE}; border: 1px solid {SAGE}55;
              border-radius: 20px; padding: 0.1rem 0.6rem; font-size: 0.7rem; }}
.pill-clay {{ background: {CLAY}22; color: {CLAY}; border: 1px solid {CLAY}55;
              border-radius: 20px; padding: 0.1rem 0.6rem; font-size: 0.7rem; }}

/* ---- upload zone ---- */
[data-testid="stFileUploader"] {{
    border: 1px dashed {SAND};
    border-radius: 2px;
    padding: 1rem;
}}

/* ---- data table ---- */
[data-testid="stDataFrame"] {{
    font-size: 0.8rem;
}}

/* ---- progress bar ---- */
.stProgress > div > div > div {{
    background-color: {SAGE};
}}
</style>
"""

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=BONE,
    font=dict(family="IBM Plex Mono, Courier New, monospace", color=SLATE, size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(gridcolor=SAND, linecolor=SAND, zeroline=False),
)

# Default y-axis style — merge into per-chart yaxis dicts rather than spreading
# into update_layout() alongside an explicit yaxis= kwarg (would cause TypeError).
PLOTLY_YAXIS = dict(gridcolor=SAND, linecolor=SAND, zeroline=False)
