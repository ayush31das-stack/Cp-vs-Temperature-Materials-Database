import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cp_calculations import calculate_cp_piecewise, build_material_curve

st.set_page_config(page_title="Cp vs Temperature Materials Database", layout="wide")

# A small amount of CSS just to tighten spacing and soften the default look.
# Kept minimal on purpose -- no custom fonts, no heavy restyling.
st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        h1 {font-weight: 700; letter-spacing: -0.5px;}
        [data-testid="stMetricValue"] {font-size: 1.1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Cp vs Temperature Materials Database")
st.caption(
    "Interactive heat-capacity database sourced from the NASA Glenn condensed-species "
    "coefficients (NASA/TP\u20142002-211556, Appendix D). Cp(T) is calculated from the "
    "7-term NASA polynomial: Cp\u00b0(T)/R = a\u00b7T\u207b\u00b2 + b\u00b7T\u207b\u00b9 + c + d\u00b7T + "
    "e\u00b7T\u00b2 + f\u00b7T\u00b3 + g\u00b7T\u2074, with R = 8.314510 J/(mol\u00b7K)."
)

DATA_FILE = "NASA_Cp_Materials_Database.xlsx"
SHEET_NAME = "NASA Condensed"

REQUIRED_COLS = {
    "Material Name",
    "Symbol",
    "Type",
    "Phase",
    "t_min (K)",
    "t_max (K)",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "Molecular Weight (g/mol)",
    "Reference Code",
    "Source / Notes",
    "NASA Species",
}


# -----------------------------
# LOAD DATABASE
# -----------------------------
@st.cache_data
def load_database(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=SHEET_NAME)
    raw = raw.rename(
        columns={
            "t_min (K)": "t_min",
            "t_max (K)": "t_max",
        }
    )
    return raw


try:
    raw_df = load_database(DATA_FILE)
except FileNotFoundError:
    st.error(f"{DATA_FILE} was not found.")
    st.stop()

missing = {c for c in REQUIRED_COLS if c not in ("t_min (K)", "t_max (K)") and c not in raw_df.columns}
if "t_min" not in raw_df.columns or "t_max" not in raw_df.columns:
    missing.add("t_min (K) / t_max (K)")
if missing:
    st.error(f"{DATA_FILE} is missing required column(s): {', '.join(sorted(missing))}")
    st.stop()

raw_df = raw_df.sort_values(["NASA Species", "t_min"]).reset_index(drop=True)

# -----------------------------
# BUILD MATERIAL-LEVEL VIEW
# -----------------------------
# NASA fits each material's Cp separately per phase/temperature interval
# (e.g. solid "AL(cr)" below 933.61 K, liquid "AL(L)" above it), each with
# its own a..g coefficients. Rather than treat every phase as a separate
# selectable item, combine every phase/interval belonging to the same
# material (keyed by Symbol, which maps 1:1 to Material Name in this
# dataset) into one continuous, piecewise-evaluated Cp(T) curve. Plotting
# it this way is what lets a phase transition (melting, a crystal
# structure change, etc.) show up as a sharp rise or fall at the
# transition temperature within a single piecewise trace.
material_records = []
segments_by_material = {}

for symbol, group in raw_df.groupby("Symbol", sort=False):
    # Use the lowest-temperature phase as the "representative" row for
    # display metadata (Reference Code, etc.). Note this can differ from
    # group.iloc[0]: rows arrive sorted by NASA Species name, and phase
    # suffixes like "(L)" can sort alphabetically before "(a)"/"(cr)",
    # so the liquid phase could otherwise get picked as "first" even
    # though it's the highest-temperature phase. Material Name/Type/MW
    # are identical across phases for every material in this dataset,
    # but Reference Code occasionally differs (e.g. B4C, H2O cite a
    # different source for their liquid vs. solid fit).
    group_sorted = group.sort_values("t_min")
    first = group_sorted.iloc[0]
    segments_by_material[symbol] = list(group_sorted.itertuples(index=False))
    phases = group_sorted["Phase"].tolist()
    transitions = sorted(
        {float(t) for t in group_sorted["t_min"]} | {float(t) for t in group_sorted["t_max"]}
    )
    # Interior boundaries only (drop the overall Tmin/Tmax) -- these are
    # the temperatures where the coefficients (and therefore the curve)
    # switch from one phase's fit to the next.
    phase_transition_temps = transitions[1:-1] if len(transitions) > 2 else []
    material_records.append(
        {
            "Symbol": symbol,
            "Material Name": first["Material Name"],
            "Category": first["Type"],
            "Phases": " \u2192 ".join(phases),
            "Tmin": float(group["t_min"].min()),
            "Tmax": float(group["t_max"].max()),
            "Molecular Weight (g/mol)": first["Molecular Weight (g/mol)"],
            "Reference Code": first["Reference Code"],
            "Intervals": len(group),
            "Phase Transitions (K)": ", ".join(f"{t:g}" for t in phase_transition_temps)
            if phase_transition_temps
            else "\u2014",
        }
    )

material_df = pd.DataFrame(material_records)
material_df["Label"] = material_df["Material Name"] + " (" + material_df["Symbol"] + ")"
label_to_symbol = dict(zip(material_df["Label"], material_df["Symbol"]))

# -----------------------------
# SIDEBAR: FILTER + SELECT
# -----------------------------
st.sidebar.header("Materials")

categories = sorted(material_df["Category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Category", options=categories, default=categories
)

filtered_df = (
    material_df[material_df["Category"].isin(selected_categories)]
    if selected_categories
    else material_df
)
st.sidebar.caption(f"{len(filtered_df)} of {len(material_df)} materials in this view")

selected_labels = st.sidebar.multiselect(
    "Search and select materials to plot",
    options=filtered_df["Label"].tolist(),
    placeholder="Type a name or symbol, e.g. 'Aluminium' or 'AL'",
)
selected_materials = [label_to_symbol[label] for label in selected_labels]

# -----------------------------
# TEMPERATURE RANGE
# -----------------------------
st.sidebar.header("Temperature range")

data_tmin = float(material_df["Tmin"].min())
data_tmax = float(material_df["Tmax"].max())
slider_lo = max(0.0, 50.0 * np.floor(data_tmin / 50.0))
slider_hi = 50.0 * np.ceil(data_tmax / 50.0)
default_range = (max(slider_lo, 300.0), min(slider_hi, 1000.0))

def sync_manual_range():
    lo, hi = st.session_state.temp_range_slider
    st.session_state.manual_tmin = float(lo)
    st.session_state.manual_tmax = float(hi)

if "manual_tmin" not in st.session_state:
    st.session_state.manual_tmin = float(default_range[0])
if "manual_tmax" not in st.session_state:
    st.session_state.manual_tmax = float(default_range[1])

st.sidebar.slider(
    "Drag to set the range (K)",
    min_value=slider_lo,
    max_value=slider_hi,
    value=default_range,
    step=5.0,
    key="temp_range_slider",
    on_change=sync_manual_range,
)

st.sidebar.markdown("**Or enter exact values (K)**")
manual_col1, manual_col2 = st.sidebar.columns(2)
with manual_col1:
    st.number_input(
        "Minimum",
        min_value=float(slider_lo),
        max_value=float(slider_hi),
        step=0.1,
        key="manual_tmin",
    )
with manual_col2:
    st.number_input(
        "Maximum",
        min_value=float(slider_lo),
        max_value=float(slider_hi),
        step=0.1,
        key="manual_tmax",
    )

Tmin_input = float(st.session_state.manual_tmin)
Tmax_input = float(st.session_state.manual_tmax)
st.sidebar.caption("Exact values are used for plotting. Moving the slider updates these values.")

# -----------------------------
# MATERIAL INFORMATION TABLE
# -----------------------------
if selected_materials:
    st.subheader("Selected materials")
    st.caption(
        "Each row combines every NASA phase/interval fitted for that material into one "
        "curve. 'Phase transitions' marks the temperature(s) where the fit switches phase "
        "(e.g. solid \u2192 liquid) -- that's where you'll see a sharp jump in the Cp plot."
    )

    info_table = material_df[material_df["Symbol"].isin(selected_materials)][
        [
            "Material Name",
            "Symbol",
            "Category",
            "Phases",
            "Tmin",
            "Tmax",
            "Phase Transitions (K)",
            "Molecular Weight (g/mol)",
            "Reference Code",
        ]
    ].rename(
        columns={
            "Tmin": "Valid range: Tmin (K)",
            "Tmax": "Valid range: Tmax (K)",
            "Reference Code": "Reference",
        }
    )
    st.dataframe(info_table, use_container_width=True, hide_index=True)

# -----------------------------
# Cp AT A SPECIFIC TEMPERATURE
# -----------------------------
if selected_materials:
    st.subheader("Cp at a specific temperature")

    point_T = st.number_input(
        "Temperature (K)", value=float(round((Tmin_input + Tmax_input) / 2)), step=5.0
    )

    rows = []
    for symbol in selected_materials:
        info = material_df[material_df["Symbol"] == symbol].iloc[0]
        cp_value = None
        if point_T > 0 and info["Tmin"] <= point_T <= info["Tmax"]:
            cp_value = calculate_cp_piecewise(point_T, segments_by_material[symbol])
        rows.append(
            {
                "Material": info["Label"],
                "Cp (J/mol\u00b7K)": f"{cp_value:.2f}" if cp_value is not None else "Out of range",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# -----------------------------
# COMPARE
# -----------------------------
COLORWAY = ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED", "#0891B2", "#DB2777", "#65A30D"]

if st.button("Compare materials", type="primary"):

    if not selected_materials:
        st.error("Select at least one material.")
    elif Tmin_input >= Tmax_input:
        st.error("Maximum temperature must be greater than minimum temperature.")
    else:
        out_of_range = []
        fig = go.Figure()
        export_frames = []

        for symbol in selected_materials:
            info = material_df[material_df["Symbol"] == symbol].iloc[0]
            label = info["Label"]

            # Plot the portion of the requested range that is actually covered
            # by this material's NASA data. If the requested range extends beyond
            # the material's valid range, clip to the overlap instead of dropping
            # the material completely.
            plot_tmin = max(Tmin_input, float(info["Tmin"]))
            plot_tmax = min(Tmax_input, float(info["Tmax"]))

            if plot_tmin >= plot_tmax:
                out_of_range.append(
                    f"**{label}**: no overlap with the requested range "
                    f"{Tmin_input:g}-{Tmax_input:g} K. Valid range: "
                    f"{info['Tmin']:g}-{info['Tmax']:g} K."
                )
                continue

            if plot_tmin > Tmin_input or plot_tmax < Tmax_input:
                out_of_range.append(
                    f"**{label}**: plotted only over its valid range "
                    f"{plot_tmin:g}-{plot_tmax:g} K (requested "
                    f"{Tmin_input:g}-{Tmax_input:g} K)."
                )

            # Sample each NASA interval independently so phase-transition
            # boundaries are preserved instead of being smeared by one shared grid.
            T, Cp = build_material_curve(
                segments_by_material[symbol],
                plot_tmin,
                plot_tmax,
                points_per_segment=100,
            )

            fig.add_trace(
                go.Scatter(
                    x=T,
                    y=Cp,
                    mode="lines",
                    name=label,
                    line=dict(width=3),
                    hovertemplate="T = %{x:.0f} K<br>Cp = %{y:.2f} J/mol\u00b7K<extra>" + label + "</extra>",
                )
            )
            export_frames.append(pd.DataFrame({"Material": label, "T (K)": T, "Cp (J/mol*K)": Cp}))

        if out_of_range:
            st.warning("Range notes:\n\n" + "\n\n".join(out_of_range))

        if len(fig.data) > 0:
            fig.update_layout(
                template="plotly_white",
                colorway=COLORWAY,
                title=dict(text="Cp vs Temperature", font=dict(size=20), x=0, xanchor="left"),
                xaxis_title="Temperature (K)",
                yaxis_title="Cp (J/mol\u00b7K)",
                hovermode="x unified",
                legend=dict(
                    title="Material",
                    orientation="h",
                    yanchor="top",
                    y=-0.25,
                    xanchor="center",
                    x=0.5,
                ),
                margin=dict(l=10, r=10, t=50, b=10),
                font=dict(family="sans-serif", size=13),
            )
            fig.update_xaxes(
                showgrid=True,
                gridcolor="#E5E7EB",
                zeroline=False,
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikedash="dot",
                spikethickness=1,
                spikecolor="#6B7280",
            )
            fig.update_yaxes(
                showgrid=True,
                gridcolor="#E5E7EB",
                zeroline=False,
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikedash="dot",
                spikethickness=1,
                spikecolor="#6B7280",
            )
            fig.update_layout(hoverdistance=100, spikedistance=1000)

            st.plotly_chart(fig, use_container_width=True)

            export_df = pd.concat(export_frames, ignore_index=True)
            st.download_button(
                "Download plotted data as CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="cp_vs_temperature.csv",
                mime="text/csv",
            )
        else:
            st.error("None of the selected materials are valid over this temperature range.")