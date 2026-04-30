import os
import uuid
import calendar
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from logic import (
    COLUMNS,
    append_trades,
    compute_metrics,
    convert_tradingview_to_mvizion,
    delete_trade_by_position,
    delete_account,
    ensure_csv_exists,
    format_date_fr,
    format_month_fr,
    get_trading_session,
    infer_mental_state,
    load_trades,
    load_accounts_from_sheet,
    save_screenshot,
    save_trade,
    upsert_account,
)


ETAT_MENTAL_COLORS = {
    "Concentre": "#60A5FA",
    "Confiant": "#22C55E",
    "Calme": "#34D399",
    "Anxieux": "#F59E0B",
    "Fatigue": "#A78BFA",
    "Euphorique": "#F472B6",
    "Frustre": "#EF4444",
    "Neutre": "#D1D5DB",
}


def performance_figure(df: pd.DataFrame) -> go.Figure:
    curve = df.copy()
    curve["Capital"] = curve["Profit"].cumsum()
    curve["DateLabel"] = curve["Date"].apply(format_date_fr)
    ticks = pd.date_range(start=curve["Date"].min(), end=curve["Date"].max(), freq="MS")
    if len(ticks) == 0:
        ticks = pd.to_datetime(curve["Date"].drop_duplicates())
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["Date"],
            y=curve["Capital"],
            mode="lines",
            line={"color": "#3B82F6", "width": 2.4},
            customdata=curve[["DateLabel"]],
            hovertemplate="Date: %{customdata[0]}<br>Capital: %{y:,.2f}$<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font={"color": "#FFFFFF"},
        margin=dict(l=12, r=12, t=12, b=12),
        height=350,
        xaxis={"tickvals": ticks, "ticktext": [format_month_fr(x) for x in ticks], "color": "#FFFFFF", "gridcolor": "#242A35"},
        yaxis={"color": "#FFFFFF", "gridcolor": "#242A35"},
        hoverlabel={"bgcolor": "#13161D", "font": {"color": "#FFFFFF"}},
        showlegend=False,
    )
    return fig


def _discipline_score(df: pd.DataFrame) -> pd.Series:
    cols = ["Sizing_Score", "SL_Score", "Revenge_Score", "Overtrading_Score", "Bias_Score"]
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=1)


def heatmap_jour_heure_profit(df: pd.DataFrame) -> go.Figure:
    """Heatmap Lun–Ven × heure : somme des PnL par créneau (contribution au cumul)."""
    day_names = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi"]
    hours = list(range(24))
    if df.empty:
        z_empty = np.zeros((5, 24))
        return go.Figure(
            data=go.Heatmap(
                z=z_empty,
                x=[f"{h:02d}h" for h in hours],
                y=day_names,
                colorscale=[[0, "#0E1117"], [1, "#6366F1"]],
                showscale=False,
            )
        ).update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font={"color": "#FFFFFF"},
            height=400,
        )

    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date"])
    work = work[work["Date"].dt.dayofweek <= 4]
    work["Profit"] = pd.to_numeric(work["Profit"], errors="coerce").fillna(0.0)

    SESSION_HOUR = {"ASIA": 3, "LONDON": 11, "NY": 18, "NEW YORK": 18, "OUT": 12}
    sess_series = work["Session"].astype(str).str.strip().str.upper().map(lambda s: SESSION_HOUR.get(s, 12))
    midnight = (
        (work["Date"].dt.hour == 0)
        & (work["Date"].dt.minute == 0)
        & (work["Date"].dt.second == 0)
    )
    work["_hour"] = work["Date"].dt.hour.clip(0, 23)
    work.loc[midnight, "_hour"] = sess_series.loc[midnight].astype(int)

    work["_day"] = work["Date"].dt.dayofweek.map(lambda i: day_names[i] if i < 5 else None)
    work = work.dropna(subset=["_day"])
    if work.empty:
        z0 = np.zeros((5, 24))
        return go.Figure(
            data=go.Heatmap(
                z=z0,
                x=[f"{h:02d}h" for h in hours],
                y=day_names,
                colorscale=[[0, "#0E1117"], [1, "#6366F1"]],
                showscale=False,
                hovertemplate="%{y} · %{x}<br>Aucune donnée<extra></extra>",
            )
        ).update_layout(
            title=dict(
                text="Profit cumulé par jour (lun–ven) et par heure",
                font=dict(color="#FFFFFF", size=15),
                x=0.02,
                xanchor="left",
            ),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font={"color": "#FFFFFF"},
            margin=dict(l=12, r=24, t=52, b=12),
            height=440,
            xaxis=dict(title="Heure", color="#A1A1AA"),
            yaxis=dict(title="", color="#A1A1AA", autorange="reversed"),
        )

    agg = work.groupby(["_day", "_hour"], as_index=False)["Profit"].sum()
    pivot = agg.pivot(index="_day", columns="_hour", values="Profit").reindex(index=day_names)
    for h in hours:
        if h not in pivot.columns:
            pivot[h] = np.nan
    pivot = pivot.reindex(columns=hours)
    z = np.nan_to_num(pivot.values.astype(float), nan=0.0)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[f"{h:02d}h" for h in hours],
            y=day_names,
            colorscale=[
                [0.0, "#0B1020"],
                [0.25, "#1e3a5f"],
                [0.5, "#4F46E5"],
                [0.75, "#818CF8"],
                [1.0, "#34D399"],
            ],
            hovertemplate="%{y} · %{x}<br>Somme des profits : %{z:,.2f} $<extra></extra>",
            colorbar=dict(
                title=dict(text="Profit ($)", side="right", font=dict(color="#E5E7EB")),
                tickfont=dict(color="#A1A1AA"),
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text="Profit cumulé par jour (lun–ven) et par heure",
            font=dict(color="#FFFFFF", size=15),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font={"color": "#FFFFFF"},
        margin=dict(l=12, r=24, t=52, b=12),
        height=440,
        xaxis=dict(title="Heure", color="#A1A1AA", gridcolor="#242A35", showgrid=False),
        yaxis=dict(title="", color="#A1A1AA", autorange="reversed"),
        hoverlabel={"bgcolor": "#13161D", "font": {"color": "#FFFFFF"}},
    )
    return fig


def discipline_profit_correlation_figure(df: pd.DataFrame) -> tuple[go.Figure, float | None]:
    """Nuage de points discipline (moyenne des 5 scores) vs profit + droite de tendance."""
    empty = go.Figure().update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font={"color": "#FFFFFF"},
        height=400,
    )
    if df.empty or len(df) < 2:
        return empty, None

    plot_df = df.copy()
    plot_df["Discipline_Score"] = _discipline_score(plot_df)
    plot_df["Profit"] = pd.to_numeric(plot_df["Profit"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Discipline_Score", "Profit"])
    x = plot_df["Discipline_Score"].to_numpy(dtype=float)
    y = plot_df["Profit"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return empty, None

    std_x, std_y = float(np.std(x)), float(np.std(y))
    if std_x < 1e-12 or std_y < 1e-12:
        corr_val: float | None = None
        coef = None
    else:
        c = np.corrcoef(x, y)[0, 1]
        corr_val = float(c) if np.isfinite(c) else None
        coef = np.polyfit(x, y, 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=11, color="#6366F1", opacity=0.72, line=dict(width=0.5, color="#A5B4FC")),
            name="Trades",
            hovertemplate="Discipline : %{x:.1f}<br>Profit : %{y:,.2f} $<extra></extra>",
        )
    )
    if coef is not None:
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 80)
        ys = np.polyval(coef, xs)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#C4B5FD", width=2.5),
                name="Tendance",
                hovertemplate="Tendance<br>Profit estimé : %{y:,.2f} $<extra></extra>",
            )
        )
    fig.update_layout(
        title=dict(
            text="Discipline vs profit — plus la moyenne des scores est haute, plus le PnL suit souvent la tendance",
            font=dict(color="#FFFFFF", size=15),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font={"color": "#FFFFFF"},
        margin=dict(l=12, r=12, t=52, b=12),
        height=420,
        xaxis=dict(
            title="Score de discipline (moyenne des 5 critères, 0–20)",
            color="#A1A1AA",
            gridcolor="#242A35",
            zeroline=False,
        ),
        yaxis=dict(title="Profit ($)", color="#A1A1AA", gridcolor="#242A35", zeroline=True, zerolinecolor="#374151"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#E5E7EB")),
        hoverlabel={"bgcolor": "#13161D", "font": {"color": "#FFFFFF"}},
        showlegend=True,
    )
    return fig, corr_val


def build_account_settings(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    if df.empty:
        return {}
    work = df.copy().sort_values("Date")
    settings: dict[str, dict[str, float]] = {}
    for _, row in work.iterrows():
        compte = str(row.get("Compte", "")).strip()
        if not compte:
            continue
        obj = pd.to_numeric(row.get("Profit_Objectif_Pct", 10.0), errors="coerce")
        mdl = pd.to_numeric(row.get("Max_Daily_Loss_USD", 500.0), errors="coerce")
        settings[compte] = {
            "profit_pct": float(obj) if pd.notna(obj) else 10.0,
            "max_daily_loss_usd": float(mdl) if pd.notna(mdl) else 500.0,
            "initial_capital": 10000.0,
        }
    return settings


def trading_activity_calendar_html(df: pd.DataFrame, year: int, month: int) -> str:
    month_mat = calendar.monthcalendar(year, month)
    day_profit = {}
    if not df.empty:
        work = df.copy()
        work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
        work["Profit"] = pd.to_numeric(work["Profit"], errors="coerce").fillna(0.0)
        work = work.dropna(subset=["Date"])
        month_df = work[(work["Date"].dt.year == year) & (work["Date"].dt.month == month)]
        if not month_df.empty:
            grp = month_df.groupby(month_df["Date"].dt.day)["Profit"].sum()
            day_profit = {int(k): float(v) for k, v in grp.items()}

    day_names = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    rows = []
    rows.append("<tr>" + "".join([f"<th>{d}</th>" for d in day_names]) + "</tr>")
    for week in month_mat:
        tds = []
        for d in week:
            if d == 0:
                tds.append('<td class="cal-empty"></td>')
            else:
                pnl = day_profit.get(d, 0.0)
                cls = "cal-neutral"
                if pnl > 0:
                    cls = "cal-green"
                elif pnl < 0:
                    cls = "cal-red"
                tds.append(f'<td class="{cls}"><div class="day">{d}</div><div class="pnl">{pnl:+.0f}$</div></td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return f"""
    <style>
        .trade-cal {{ width: 100%; border-collapse: separate; border-spacing: 6px; table-layout: fixed; }}
        .trade-cal th {{ color:#A1A1AA; font-size:0.78rem; font-weight:700; text-align:center; padding-bottom:4px; }}
        .trade-cal td {{ border:1px solid #1F1F24; border-radius:10px; background:#0F1118; height:72px; vertical-align:top; padding:6px; }}
        .trade-cal td .day {{ font-weight:700; color:#E5E7EB; font-size:0.86rem; }}
        .trade-cal td .pnl {{ margin-top:6px; font-size:0.76rem; color:#C7CEDA; }}
        .trade-cal td.cal-green {{ background:#0E2A1D; border-color:#166534; }}
        .trade-cal td.cal-green .pnl {{ color:#86EFAC; }}
        .trade-cal td.cal-red {{ background:#321015; border-color:#7F1D1D; }}
        .trade-cal td.cal-red .pnl {{ color:#FCA5A5; }}
        .trade-cal td.cal-empty {{ border:none; background:transparent; }}
    </style>
    <table class="trade-cal">
        {''.join(rows)}
    </table>
    """


st.set_page_config(page_title="Mat'Sa", layout="wide")
ensure_csv_exists()

st.markdown(
    """
    <style>
        .stApp { background: #050505; color: #FFFFFF; font-family: "Inter", sans-serif; }
        [data-testid="stSidebar"] { background: #0D1117; border-right: 1px solid #1f2937; padding-top: 2rem; }
        [data-testid="stSidebar"] * { font-size: 1.02rem !important; font-weight: 620 !important; color: #F3F4F6; }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
            color: #F9FAFB !important;
            letter-spacing: 0.2px;
        }
        .block-container { padding-top: 2.35rem; padding-bottom: 1.15rem; max-width: 1460px; }
        .tv-logo {
            position: relative;
            display: block;
            text-align: center;
            font-size: 48px;
            font-weight: 700;
            font-style: italic;
            color: #D4AF37;
            margin: 0.2rem 0 1.2rem 0;
            padding: 0.25rem 0.15rem 0.35rem 0.15rem;
            letter-spacing: 0.5px;
            line-height: 1.05;
            font-family: "Cormorant Garamond", "Palatino Linotype", "Times New Roman", serif;
            overflow: hidden;
            text-shadow: 0 1px 0 rgba(255, 255, 255, 0.12), 0 8px 24px rgba(212, 175, 55, 0.22);
        }
        .tv-logo::after {
            content: "";
            position: absolute;
            top: -12%;
            left: -45%;
            width: 38%;
            height: 130%;
            background: linear-gradient(
                110deg,
                rgba(255, 255, 255, 0.0) 0%,
                rgba(255, 248, 220, 0.65) 45%,
                rgba(255, 255, 255, 0.0) 100%
            );
            transform: skewX(-22deg);
            animation: logo-shimmer 3.4s ease-in-out infinite;
            pointer-events: none;
        }
        @keyframes logo-shimmer {
            0% { left: -52%; }
            55% { left: 112%; }
            100% { left: 112%; }
        }
        .tv-card {
            background: linear-gradient(180deg, #121218 0%, #101015 100%);
            border: 1px solid #1F1F24;
            border-radius: 12px;
            padding: 14px 14px 12px 14px;
            min-height: 112px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.28);
        }
        .tv-card-profit { border: 1px solid #1F1F24; }
        .tv-title {
            color: #A1A1AA;
            font-size: 0.69rem;
            margin-top: 7px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.55px;
            line-height: 1.15;
        }
        .tv-value {
            color: #FFFFFF;
            font-size: 1.62rem;
            font-weight: 800;
            font-family: "Inter", sans-serif;
            letter-spacing: 0.15px;
            line-height: 1.05;
        }
        .pnl-glow { color: #00FFA3; text-shadow: 0 0 14px rgba(0,255,163,0.28), 0 0 24px rgba(0,255,163,0.14); }
        .tvs-badge {
            background: radial-gradient(circle at 30% 30%, rgba(99,102,241,0.46), rgba(99,102,241,0.16));
            border: 1px solid #6366F1;
            border-radius: 999px;
            width: 92px;
            height: 92px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 2px auto 0 auto;
            color: #FFFFFF;
            font-size: 1.45rem;
            font-weight: 800;
            box-shadow: 0 0 22px rgba(99,102,241,0.2), inset 0 0 12px rgba(99,102,241,0.15);
        }
        .risk-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid transparent;
            line-height: 1;
            vertical-align: middle;
        }
        .risk-green { background: #0F2E1D; color: #86EFAC; border-color: #166534; }
        .risk-orange { background: #3A2308; color: #FBBF24; border-color: #B45309; }
        .risk-red { background: #3A0D11; color: #FCA5A5; border-color: #B91C1C; animation: blink-risk 1s infinite; }
        .elite-topbar {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 8px;
            padding: 5px 10px;
            color: #E5E7EB;
            font-size: 0.8rem;
            margin-bottom: 0.6rem;
            white-space: nowrap;
            overflow-x: auto;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .elite-num {
            font-family: "Consolas", "SFMono-Regular", "Roboto Mono", monospace;
            font-weight: 700;
            color: #FFFFFF;
        }
        @keyframes blink-risk {
            0% { opacity: 1; }
            50% { opacity: 0.45; }
            100% { opacity: 1; }
        }
        .stFormSubmitButton > button {
            background: linear-gradient(180deg, #6366F1 0%, #4F46E5 100%);
            border: 1px solid #1E40AF; color: #FFFFFF; width: 100%; min-height: 44px; border-radius: 8px; font-weight: 700;
            box-shadow: 0 8px 18px rgba(79, 70, 229, 0.3);
            letter-spacing: 0.2px;
        }
        .stFormSubmitButton > button:hover {
            background: linear-gradient(180deg, #7376FA 0%, #5B50EE 100%);
            border-color: #6366F1;
        }
        [data-testid="stHorizontalBlock"] { gap: 0.72rem; }
        .stPlotlyChart > div { border-radius: 12px; }
        h3 { margin-top: 0.45rem; margin-bottom: 0.5rem; }
        .stMarkdown p { margin-bottom: 0.45rem; }
        .stProgress > div > div > div > div { background-color: #6366F1; }
        .stDataFrame { border: 1px solid #1F1F24; border-radius: 12px; overflow: hidden; }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #FFFFFF !important; }
        [data-testid="stSelectbox"] > div, [data-testid="stNumberInput"] > div {
            border-color: #1F1F24 !important;
            border-radius: 10px !important;
        }
        /* Premium Gallery (captures trade — seul usage st.image) */
        [data-testid="stImage"] {
            display: flex !important;
            justify-content: center !important;
            width: 100%;
        }
        [data-testid="stImage"] img,
        [data-testid="stImage"] picture img {
            border-radius: 12px !important;
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.55), 0 4px 14px rgba(0, 0, 0, 0.35) !important;
            max-width: min(920px, 100%) !important;
            width: auto !important;
            object-fit: contain;
        }
        .session-chart-wrap {
            margin: 0.8rem 0 1.1rem 0;
            padding: 0.7rem;
            border-radius: 12px;
            border: 1px solid #1F1F24;
            background: #090B10;
        }
        .settings-card {
            border: 1px solid #1F1F24;
            border-radius: 12px;
            padding: 14px;
            margin: 0.55rem 0 0.95rem 0;
            background: linear-gradient(180deg, #11141C 0%, #0C0F15 100%);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

all_trades = load_trades()
sheet_accounts = load_accounts_from_sheet()
account_settings = build_account_settings(all_trades)
for _, row in sheet_accounts.iterrows():
    account_settings[str(row["Nom"]).strip()] = {
        "profit_pct": float(pd.to_numeric(row["Objectif_Pct"], errors="coerce") if pd.notna(row["Objectif_Pct"]) else 10.0),
        "max_daily_loss_usd": float(pd.to_numeric(row["Max_Loss_USD"], errors="coerce") if pd.notna(row["Max_Loss_USD"]) else 500.0),
        "initial_capital": float(pd.to_numeric(row.get("Initial_Capital", 10000.0), errors="coerce") if pd.notna(row.get("Initial_Capital", 10000.0)) else 10000.0),
    }
trade_accounts = sorted([x for x in all_trades["Compte"].astype(str).unique().tolist() if x]) if not all_trades.empty else []
empty_accounts = [str(x).strip() for x in sheet_accounts["Nom"].astype(str).tolist() if str(x).strip()]
account_names = sorted(set(trade_accounts + empty_accounts))
if "pending_trade_compte" not in st.session_state:
    st.session_state.pending_trade_compte = ""

with st.sidebar:
    st.sidebar.markdown("<div class=\"tv-logo\">Mat'Sa</div>", unsafe_allow_html=True)

    compte_options = ["Tous les comptes"]
    if account_names:
        compte_options += account_names
    selected_compte = st.selectbox("Sélectionner un Compte", compte_options, key="global_compte_filter")

    with st.expander("Paramètres du Compte", expanded=False):
        selected_settings = account_settings.get(selected_compte, {"profit_pct": 10.0, "max_daily_loss_usd": 500.0})
        objectif_profit_pct = st.number_input("Objectif de Profit (%)", min_value=1.0, max_value=100.0, value=float(selected_settings["profit_pct"]), step=0.5, key="objectif_profit_pct")
        max_daily_loss_usd = st.number_input("Max Daily Loss ($)", min_value=1.0, value=float(selected_settings["max_daily_loss_usd"]), step=25.0, key="max_daily_loss_usd")

    components.html(
        """
        <div style="display:block;padding-right:8px;">
            <div style="background:#151922;border:1px solid #2F3645;border-radius:8px;padding:5px 8px;margin-bottom:8px;">
                <div style="font-size:10px;color:#6B7280;font-weight:700;text-transform:uppercase;">New York</div>
                <div id="ny_time" style="font-family:'Courier New',monospace;color:#FFFFFF;font-size:20px;font-weight:700;">--:--:--</div>
            </div>
            <div style="background:#151922;border:1px solid #2F3645;border-radius:8px;padding:5px 8px;">
                <div style="font-size:10px;color:#6B7280;font-weight:700;text-transform:uppercase;">Paris</div>
                <div id="paris_time" style="font-family:'Courier New',monospace;color:#FFFFFF;font-size:20px;font-weight:700;">--:--:--</div>
            </div>
        </div>
        <script>
            function updateClocks() {
                const ny = new Date().toLocaleTimeString('fr-FR', {timeZone:'America/New_York', hour12:false});
                const paris = new Date().toLocaleTimeString('fr-FR', {timeZone:'Europe/Paris', hour12:false});
                document.getElementById('ny_time').innerText = ny;
                document.getElementById('paris_time').innerText = paris;
            }
            updateClocks();
            setInterval(updateClocks, 1000);
        </script>
        """,
        height=120,
    )
    page = st.radio(
        "Navigation",
        ["Dashboard", "📅 Calendrier", "Mes Stats", "Analyses Avancées", "Mon Trading", "Mon Compte/Finance", "Nouveau Trade", "⚙️ Paramètres"],
        key="main_nav",
    )
    st.markdown("---")
    st.header("Importer depuis TradingView")
    uploaded_file = st.file_uploader(
        "Importer un fichier TradingView (1GB per file)",
        type=["csv"],
        key="tv_import",
        help="1GB per file",
    )
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except Exception:
            try:
                uploaded_file.seek(0)
                raw_df = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="utf-8-sig", on_bad_lines="skip")
            except Exception:
                uploaded_file.seek(0)
                raw_df = pd.read_csv(uploaded_file, sep=";", encoding="utf-8-sig", on_bad_lines="skip")
        try:
            converted = convert_tradingview_to_mvizion(raw_df)
            if converted.empty:
                st.error("Aucun trade valide trouve dans le fichier.")
            else:
                st.success(f"Importation de {append_trades(converted)} trades reussie.")
                st.rerun()
        except Exception:
            st.error("Echec importation TradingView. Verifier le CSV.")

if selected_compte == "Tous les comptes":
    trades = all_trades.copy()
else:
    trades = all_trades[all_trades["Compte"].astype(str) == selected_compte].copy()

if page == "Nouveau Trade":
    st.subheader("Nouveau Trade")
    with st.form("trade_form", clear_on_submit=True):
        top_left, top_right = st.columns([1.3, 1.1])
        with top_left:
            trade_date = st.date_input("Date", value=date.today(), key="trade_date")
            a1, a2 = st.columns([1.4, 1.0])
            with a1:
                actif = st.selectbox(
                    "Actif",
                    ["MNQ1!", "MES1!", "DXY", "MGC1!", "NAS100"],
                    key="trade_actif",
                )
            with a2:
                trade_type = st.radio("Type", ["BUY", "SELL"], horizontal=True, key="trade_type")
            trade_account_options = account_names.copy() if account_names else ["Compte 1"]
            add_account_label = "➕ Ajouter un compte"
            if add_account_label not in trade_account_options:
                trade_account_options.append(add_account_label)
            pending = str(st.session_state.get("pending_trade_compte", "")).strip()
            selected_index = trade_account_options.index(pending) if pending in trade_account_options else 0
            compte = st.selectbox("Compte", trade_account_options, index=selected_index, key="trade_compte")
            if pending:
                st.session_state.pending_trade_compte = ""
            if compte == add_account_label:
                st.info("Crée un compte ci-dessous puis ajoute le trade.")
            compte_type = st.selectbox("Type de Compte", ["Evaluation", "Funded", "Live"], key="trade_compte_type")
            session = st.selectbox("Session", ["AUTO", "ASIA", "LONDON", "NEW YORK", "OUT"], index=0, key="trade_session")
            default_acc = account_settings.get(compte, {"profit_pct": 10.0, "max_daily_loss_usd": 500.0})
            profit_objectif_pct = st.number_input(
                "% de Profit Objectif (compte)",
                min_value=1.0,
                max_value=100.0,
                value=float(default_acc["profit_pct"]),
                step=0.5,
                key="trade_profit_objectif_pct",
            )
            max_daily_loss_usd_trade = st.number_input(
                "Max Daily Loss (en $)",
                min_value=1.0,
                value=float(default_acc["max_daily_loss_usd"]),
                step=25.0,
                key="trade_max_daily_loss_usd",
            )
            with st.expander("➕ Ajouter un compte", expanded=False):
                new_account_name = st.text_input("Nom du nouveau compte", key="new_account_name")
                new_account_profit_pct = st.number_input(
                    "% de Profit Objectif",
                    min_value=1.0,
                    max_value=100.0,
                    value=10.0,
                    step=0.5,
                    key="new_account_profit_pct",
                )
                new_account_max_loss = st.number_input(
                    "Max Daily Loss (en $)",
                    min_value=1.0,
                    value=500.0,
                    step=25.0,
                    key="new_account_max_loss",
                )
                add_account_submit = st.form_submit_button("Créer ce compte")
                if add_account_submit:
                    clean_name = new_account_name.strip()
                    if not clean_name:
                        st.warning("Le nom du compte est obligatoire.")
                    else:
                        upsert_account(clean_name, float(new_account_profit_pct), float(new_account_max_loss), 10000.0)
                        account_settings[clean_name] = {
                            "profit_pct": float(new_account_profit_pct),
                            "max_daily_loss_usd": float(new_account_max_loss),
                        }
                        st.session_state.pending_trade_compte = clean_name
                        st.success(f"Compte '{clean_name}' ajouté.")
                        st.rerun()
        with top_right:
            prix_entree = st.number_input("Prix Entree", min_value=0.0, value=0.0, step=0.01, key="trade_prix_entree")
            prix_tp = st.number_input("TP", min_value=0.0, value=0.0, step=0.01, key="trade_prix_tp")
            prix_sl = st.number_input("Prix du SL", min_value=0.0, value=0.0, step=0.01, key="trade_prix_sl")
            quantite = st.number_input("Quantite", min_value=0.0, value=0.0, step=0.01, key="trade_quantite")
            frais = st.number_input("Frais", min_value=0.0, value=0.0, step=0.01, key="trade_frais")
            sortie = st.selectbox("Sortie", ["SL", "TP", "BE", "TP Partiel"], key="trade_sortie")
        st.markdown("### Score execution (0-20)")
        s1, s2 = st.columns(2)
        with s1:
            sizing_score = st.slider("Sizing", 0, 20, 10, key="slider_sizing")
            sl_score = st.slider("Gestion SL", 0, 20, 10, key="slider_sl")
            revenge_score = st.slider("Controle Revenge", 0, 20, 0, key="slider_revenge")
        with s2:
            overtrading_score = st.slider("Over-trading", 0, 20, 0, key="slider_overtrading")
            bias_score = st.slider("Coherence Biais", 0, 20, 10, key="slider_bias")
            high_water_mark = st.number_input("High_Water_Mark", min_value=0.0, value=0.0, step=10.0, key="trade_high_water_mark")
        trade_screenshot = st.file_uploader(
            "Capture d'écran du graphique (1GB per file)",
            type=["png", "jpg", "jpeg"],
            key="trade_graph_screenshot",
            help="1GB per file",
        )
        submit = st.form_submit_button("Ajouter Trade", use_container_width=True)
    if submit:
        if compte == "➕ Ajouter un compte":
            st.error("Sélectionne un compte valide ou crée-en un avant d'ajouter le trade.")
        elif quantite <= 0:
            st.error("La quantite doit etre superieure a zero.")
        else:
            profit = (prix_tp - prix_entree) * quantite - frais
            etat_mental = infer_mental_state(float(sizing_score), float(sl_score), float(revenge_score), float(overtrading_score), float(bias_score))
            trade_id = f"{trade_date.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}"
            image_rel = save_screenshot(trade_screenshot, trade_id) if trade_screenshot else ""
            existing_cap = float(account_settings.get(compte, {}).get("initial_capital", 10000.0))
            upsert_account(compte, float(profit_objectif_pct), float(max_daily_loss_usd_trade), existing_cap)
            save_trade(
                {
                    "Date": pd.to_datetime(trade_date).strftime("%Y-%m-%d"),
                    "Actif": str(actif).strip().upper(),
                    "Type": trade_type,
                    "Prix Entree": float(prix_entree),
                    "Prix Sortie": float(prix_tp),
                    "Quantite": float(quantite),
                    "Frais": float(frais),
                    "Profit": float(profit),
                    "Sortie": sortie,
                    "Session": get_trading_session() if session == "AUTO" else session,
                    "Etat Mental": etat_mental,
                    "Biais Jour": "Haussier" if bias_score >= 10 else "Baissier",
                    "Compte": compte,
                    "Compte_Type": "Eval" if compte_type == "Evaluation" else compte_type,
                    "Profit_Objectif_Pct": float(profit_objectif_pct),
                    "Max_Daily_Loss_USD": float(max_daily_loss_usd_trade),
                    "Sizing_Score": float(sizing_score),
                    "SL_Score": float(sl_score),
                    "Revenge_Score": float(revenge_score),
                    "Overtrading_Score": float(overtrading_score),
                    "Bias_Score": float(bias_score),
                    "High_Water_Mark": float(high_water_mark),
                    "Image": image_rel,
                }
            )
            st.success("Trade enregistre avec succes.")
            st.rerun()

elif page == "Dashboard":
    st.subheader("Dashboard - Vue d ensemble")
    m = compute_metrics(trades)
    capital_initial = float(st.session_state.get("capital_initial", 10000.0))
    if capital_initial <= 0:
        capital_initial = 10000.0

    # Topbar Prop-Firm
    open_mask = ~trades["Sortie"].isin(["SL", "TP", "TP Partiel"]) if not trades.empty else pd.Series([], dtype=bool)
    if not trades.empty and open_mask.any():
        exposure_total = float((trades.loc[open_mask, "Prix Entree"] * trades.loc[open_mask, "Quantite"]).abs().sum())
        exposition_text = f"${exposure_total:,.2f}"
    else:
        exposition_text = "Pret a trader"

    drawdown_pct = float(m["drawdown_pct"])
    if drawdown_pct > 4:
        risk_label = "Alerte Rouge"
        risk_class = "risk-red"
    elif drawdown_pct >= 3:
        risk_label = "Alerte Orange"
        risk_class = "risk-orange"
    else:
        risk_label = "Risque Stable"
        risk_class = "risk-green"

    st.markdown(
        f"""
        <div class="elite-topbar">
            💼 Exposition Totale: <span class="elite-num">{exposition_text}</span>
            &nbsp; | &nbsp;
            ⚠️ Alerte Risque: <span class="risk-badge {risk_class}">{risk_label} ({drawdown_pct:.2f}%)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sharpe_color = "#22C55E" if m["sharpe_ratio"] > 1 else "#FFFFFF"
    drawdown_color = "#EF4444" if m["drawdown_pct"] > 3 else "#FFFFFF"
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.markdown(f'<div class="tv-card tv-card-profit"><div class="tv-value pnl-glow">${m["net_pnl"]:,.2f}</div><div class="tv-title">P&L Net</div></div>', unsafe_allow_html=True)
    r1c2.markdown(f'<div class="tv-card"><div class="tv-value">{m["win_rate"]:.2f}%</div><div class="tv-title">Win Rate</div></div>', unsafe_allow_html=True)
    r1c3.markdown(f'<div class="tv-card"><div class="tv-value">{m["profit_factor"]:.2f}</div><div class="tv-title">Profit Factor</div></div>', unsafe_allow_html=True)
    r1c4.markdown(f'<div class="tv-card"><div class="tvs-badge">{m["tvs_score"]:.0f}</div><div class="tv-title" style="text-align:center;">TVS Score</div></div>', unsafe_allow_html=True)
    r2c1, r2c2 = st.columns(2)
    r2c1.markdown(f'<div class="tv-card"><div class="tv-value">{m["avg_rr_reel"]:.2f}</div><div class="tv-title">Avg R:R Reel</div></div>', unsafe_allow_html=True)
    r2c2.markdown(f'<div class="tv-card"><div class="tv-value" style="color:{sharpe_color};">{m["sharpe_ratio"]:.2f}</div><div class="tv-title">Sharpe Ratio</div></div>', unsafe_allow_html=True)
    r3c1, r3c2 = st.columns(2)
    r3c1.markdown(f'<div class="tv-card"><div class="tv-title">Score Discipline Moyen</div><div class="tv-value">{m["discipline_score_moyen"]:.2f}/20</div></div>', unsafe_allow_html=True)
    r3c2.markdown(f'<div class="tv-card"><div class="tv-title">Drawdown Actuel</div><div class="tv-value" style="color:{drawdown_color};">${m["drawdown_actuel"]:,.2f} ({m["drawdown_pct"]:.2f}%)</div></div>', unsafe_allow_html=True)
    st.write(f"TP: {m['sorties_tp']} | TP Partiel: {m['sorties_tp_partiel']} | SL: {m['sorties_sl']}")
    session_profit = m["profit_par_session"]
    session_fig = go.Figure(
        data=[
            go.Bar(
                x=["ASIA", "LONDON", "NY"],
                y=[session_profit["ASIA"], session_profit["LONDON"], session_profit["NY"]],
                marker_color=[
                    "#22C55E" if session_profit["ASIA"] >= 0 else "#EF4444",
                    "#22C55E" if session_profit["LONDON"] >= 0 else "#EF4444",
                    "#22C55E" if session_profit["NY"] >= 0 else "#EF4444",
                ],
            )
        ]
    )
    session_fig.update_layout(
        paper_bgcolor="#050505",
        plot_bgcolor="#050505",
        font={"color": "#FFFFFF"},
        height=260,
        margin=dict(l=10, r=10, t=20, b=10),
        title="Profit par session",
    )
    st.markdown('<div class="session-chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(session_fig, use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

    # Objectifs d evaluation
    st.markdown("### Objectifs d Evaluation")
    profit_target_value = capital_initial * (float(objectif_profit_pct) / 100.0)
    progress_profit = 0.0 if profit_target_value <= 0 else max(0.0, min(1.0, m["net_pnl"] / profit_target_value))
    st.write(f"Profit Target: {m['net_pnl']:,.2f}$ / {profit_target_value:,.2f}$")
    st.progress(progress_profit)

    if trades.empty:
        daily_pnl = 0.0
    else:
        today = pd.Timestamp.now().normalize()
        today_df = trades[trades["Date"].dt.normalize() == today]
        daily_pnl = float(today_df["Profit"].sum()) if not today_df.empty else 0.0
    max_daily_loss_value = float(max_daily_loss_usd)
    used_daily_loss = abs(min(0.0, daily_pnl))
    progress_daily_loss = 0.0 if max_daily_loss_value <= 0 else max(0.0, min(1.0, used_daily_loss / max_daily_loss_value))
    st.write(f"Max Daily Loss: {used_daily_loss:,.2f}$ / {max_daily_loss_value:,.2f}$")
    st.progress(progress_daily_loss)

    # Finance globale
    st.markdown("### Finance Globale")
    roi_global = (m["net_pnl"] / capital_initial) * 100.0 if capital_initial > 0 else 0.0
    if trades.empty:
        payout_estime = 0.0
    else:
        funded_mask = trades["Compte_Type"].astype(str).str.upper().eq("FUNDED")
        funded_profit = float(trades.loc[funded_mask, "Profit"].sum()) if funded_mask.any() else 0.0
        payout_estime = max(0.0, funded_profit * 0.80)
    f1, f2 = st.columns(2)
    f1.markdown(f'<div class="tv-card"><div class="tv-title">ROI Global (%)</div><div class="tv-value">{roi_global:.2f}%</div></div>', unsafe_allow_html=True)
    f2.markdown(f'<div class="tv-card"><div class="tv-title">Payouts estimes</div><div class="tv-value">${payout_estime:,.2f}</div></div>', unsafe_allow_html=True)

    if trades.empty:
        st.info("Aucun trade pour afficher la performance.")
    else:
        st.plotly_chart(performance_figure(trades), use_container_width=True, theme=None)

elif page == "📅 Calendrier":
    st.subheader("📅 Calendrier")
    st.markdown("### Suivi d'Activité")
    now_dt = pd.Timestamp.now()
    cal_html = trading_activity_calendar_html(trades, int(now_dt.year), int(now_dt.month))
    st.markdown(cal_html, unsafe_allow_html=True)

    st.markdown("### News économiques - Investing.com (FR)")
    st.caption("Filtre US à fort impact.")
    components.html(
        """
        <iframe
            src="https://fr.investing.com/economic-calendar/"
            width="100%"
            height="680"
            frameborder="0"
            style="border:1px solid #1F1F24;border-radius:12px;background:#0E1117;">
        </iframe>
        """,
        height=700,
    )
    st.markdown("### News économiques - Forex Factory")
    st.caption("Source complémentaire (US impact fort).")
    components.html(
        """
        <iframe
            src="https://www.dailyfx.com/economic-calendar"
            width="100%"
            height="520"
            frameborder="0"
            style="border:1px solid #1F1F24;border-radius:12px;background:#0E1117;">
        </iframe>
        """,
        height=540,
    )

elif page == "Mes Stats":
    st.subheader("Mes Stats - Statistiques detaillees")
    if trades.empty:
        st.info("Aucune statistique disponible.")
    else:
        mental_stats = trades.groupby("Etat Mental", as_index=False)["Profit"].mean().sort_values("Profit", ascending=False)
        mental_stats["Couleur"] = mental_stats["Etat Mental"].map(ETAT_MENTAL_COLORS).fillna("#D1D5DB")
        fig_mental = go.Figure(data=[go.Bar(x=mental_stats["Etat Mental"], y=mental_stats["Profit"], marker_color=mental_stats["Couleur"])])
        fig_mental.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font={"color": "#FFFFFF"}, height=350)
        st.plotly_chart(fig_mental, use_container_width=True, theme=None)
        st.markdown("### Analyse rapide par session")
        m_stats = compute_metrics(trades)
        table_df = pd.DataFrame(
            {
                "Session": ["ASIA", "LONDON", "NY"],
                "Profit Total": [
                    m_stats["profit_par_session"]["ASIA"],
                    m_stats["profit_par_session"]["LONDON"],
                    m_stats["profit_par_session"]["NY"],
                ],
                "Winrate (%)": [
                    m_stats["winrate_par_session"]["ASIA"],
                    m_stats["winrate_par_session"]["LONDON"],
                    m_stats["winrate_par_session"]["NY"],
                ],
            }
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)

elif page == "Analyses Avancées":
    st.subheader("Analyses Avancées")
    if trades.empty:
        st.info("Aucun trade à analyser pour le compte sélectionné.")
    else:
        st.markdown("### Carte de chaleur : profit par jour et heure")
        st.caption(
            "Chaque case = somme des profits des trades dans ce créneau (contribution au PnL cumulé). "
            "Jours : lundi → vendredi uniquement. "
            "Si la date n’indique qu’un jour (sans heure), l’heure est estimée à partir de la session (repère Europe/Paris, comme le journal)."
        )
        fig_heat = heatmap_jour_heure_profit(trades)
        st.plotly_chart(fig_heat, use_container_width=True, theme=None)

        st.markdown("### Lien entre discipline et résultat")
        st.caption(
            "Le score de discipline est la moyenne des cinq critères d’exécution (0–20). "
            "La ligne violette est une tendance statistique : elle aide à visualiser si les trades plus disciplinés coïncident avec de meilleurs profits."
        )
        fig_disc, corr = discipline_profit_correlation_figure(trades)
        st.plotly_chart(fig_disc, use_container_width=True, theme=None)
        if corr is not None:
            adj = "positive" if corr > 0 else "négative"
            st.markdown(
                f'<p style="color:#A1A1AA;font-size:0.95rem;margin-top:0.35rem;">'
                f"Corrélation de Pearson (discipline vs profit) : <strong style=\"color:#E5E7EB;\">{corr:+.2f}</strong> "
                f"— tendance <strong style=\"color:#E5E7EB;\">{adj}</strong>.</p>",
                unsafe_allow_html=True,
            )
        elif len(trades) >= 2:
            st.caption("Corrélation non calculable (scores ou profits identiques sur tous les trades affichés).")

elif page == "Mon Trading":
    st.subheader("Mon Trading - Journal complet")
    if trades.empty:
        st.info("Aucun trade enregistre.")
    else:
        sorted_df = trades.sort_values("Date", ascending=False).reset_index()
        options = [f"{format_date_fr(r['Date'])} | {r['Actif']} | ${r['Profit']:.2f}" for _, r in sorted_df.iterrows()]
        selected = st.selectbox(
            "Sélectionner un trade (aperçu graphique / suppression)",
            options=options,
            key="delete_trade_select",
        )
        sel_pos = options.index(selected)
        selected_row = sorted_df.iloc[sel_pos]
        selected_index = int(selected_row["index"])
        img_raw = str(selected_row.get("Image", "") or "").strip()
        if img_raw and os.path.isfile(img_raw):
            _, col_img, _ = st.columns([0.12, 0.76, 0.12])
            with col_img:
                st.image(img_raw, use_container_width=True)
        else:
            st.markdown(
                '<p style="text-align:center;color:#A1A1AA;font-size:1rem;margin:0.75rem 0 1rem 0;">'
                "📷 Aucun graphique enregistré pour ce trade"
                "</p>",
                unsafe_allow_html=True,
            )
        if st.button("Supprimer ce trade"):
            delete_trade_by_position(selected_index)
            st.success("Trade supprime.")
            st.rerun()
        display_df = trades.sort_values("Date", ascending=False).copy()
        display_df["Date"] = display_df["Date"].apply(format_date_fr)
        ordered_cols = ["Date", "Actif", "Session", "Type", "Prix Entree", "Prix Sortie", "Quantite", "Frais", "Profit", "Sortie", "Compte", "Compte_Type"]
        st.dataframe(display_df[ordered_cols], use_container_width=True, hide_index=True)

elif page == "⚙️ Paramètres":
    st.subheader("⚙️ Paramètres")
    st.caption("Gère les comptes enregistrés dans la feuille Accounts : consultation, mise à jour des limites et suppression.")

    accounts_df = load_accounts_from_sheet()
    st.markdown('<div class="settings-card">', unsafe_allow_html=True)
    st.markdown("#### 📋 Liste des comptes")
    if accounts_df.empty:
        st.info("Aucun compte enregistré dans la feuille Accounts.")
    else:
        display_accounts = accounts_df.rename(
            columns={
                "Nom": "Compte",
                "Objectif_Pct": "Objectif Profit (%)",
                "Max_Loss_USD": "Max Loss ($)",
            }
        )
        st.dataframe(display_accounts, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="settings-card">', unsafe_allow_html=True)
    st.markdown("#### ✏️ Modifier un compte")
    if accounts_df.empty:
        st.caption("Ajoute d'abord un compte depuis Nouveau Trade.")
    else:
        names = accounts_df["Nom"].astype(str).tolist()
        edit_name = st.selectbox("Compte à modifier", names, key="settings_edit_account")
        edit_row = accounts_df[accounts_df["Nom"].astype(str) == edit_name].iloc[-1]
        edit_obj = st.number_input(
            "Objectif Profit (%)",
            min_value=1.0,
            max_value=100.0,
            value=float(edit_row["Objectif_Pct"]),
            step=0.5,
            key="settings_edit_obj",
        )
        edit_loss = st.number_input(
            "Max Loss ($)",
            min_value=1.0,
            value=float(edit_row["Max_Loss_USD"]),
            step=25.0,
            key="settings_edit_loss",
        )
        if st.button("💾 Enregistrer les modifications", key="settings_save_account"):
            existing_cap = float(accounts_df[accounts_df["Nom"].astype(str) == edit_name]["Initial_Capital"].iloc[-1]) if "Initial_Capital" in accounts_df.columns else 10000.0
            upsert_account(edit_name, float(edit_obj), float(edit_loss), existing_cap)
            st.success(f"Compte '{edit_name}' mis à jour.")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="settings-card">', unsafe_allow_html=True)
    st.markdown("#### 🗑️ Supprimer un compte")
    if accounts_df.empty:
        st.caption("Aucun compte à supprimer.")
    else:
        names = accounts_df["Nom"].astype(str).tolist()
        del_name = st.selectbox("Compte à supprimer", names, key="settings_delete_account")
        confirm = st.checkbox(f"Confirmer la suppression définitive de '{del_name}'", key="settings_confirm_delete")
        if st.button("🧨 Supprimer ce compte", key="settings_delete_btn"):
            if not confirm:
                st.warning("Coche la confirmation avant de supprimer.")
            else:
                deleted = delete_account(del_name)
                if deleted:
                    st.success(f"Compte '{del_name}' supprimé de la feuille Accounts.")
                    st.rerun()
                else:
                    st.info("Compte introuvable ou déjà supprimé.")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Mon Compte/Finance":
    st.subheader("Mon Compte/Finance - Gestion du capital")
    if not account_names:
        st.info("Aucun compte disponible. Crée d'abord un compte depuis Nouveau Trade.")
    else:
        finance_compte = selected_compte if selected_compte in account_names else account_names[0]
        if selected_compte == "Tous les comptes":
            st.caption("Sélectionne un compte précis dans la sidebar pour des chiffres totalement isolés par compte.")
        finance_settings = account_settings.get(finance_compte, {"profit_pct": 10.0, "max_daily_loss_usd": 500.0, "initial_capital": 10000.0})
        finance_initial = st.number_input(
            "Capital initial ($)",
            min_value=0.0,
            value=float(finance_settings.get("initial_capital", 10000.0)),
            step=100.0,
            key="capital_initial_input",
        )
        if st.button("💾 Enregistrer le capital du compte", key="save_capital_finance"):
            upsert_account(
                finance_compte,
                float(finance_settings.get("profit_pct", 10.0)),
                float(finance_settings.get("max_daily_loss_usd", 500.0)),
                float(finance_initial),
            )
            st.success(f"Capital initial mis à jour pour {finance_compte}.")
            st.rerun()

        trades_finance = all_trades[all_trades["Compte"].astype(str) == finance_compte].copy()
        m = compute_metrics(trades_finance)
        capital_actuel = float(finance_initial) + float(m["net_pnl"])
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="tv-card"><div class="tv-title">Capital initial ({finance_compte})</div><div class="tv-value">${finance_initial:,.2f}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="tv-card tv-card-profit"><div class="tv-title">Capital actuel ({finance_compte})</div><div class="tv-value">${capital_actuel:,.2f}</div></div>', unsafe_allow_html=True)

        st.markdown("### Synthèse financière du compte sélectionné")
        g1, g2, g3 = st.columns(3)
        g1.markdown(f'<div class="tv-card"><div class="tv-title">Balance</div><div class="tv-value">${capital_actuel:,.2f}</div></div>', unsafe_allow_html=True)
        g2.markdown(f'<div class="tv-card"><div class="tv-title">Profit total</div><div class="tv-value">${m["net_pnl"]:,.2f}</div></div>', unsafe_allow_html=True)
        g3.markdown(f'<div class="tv-card"><div class="tv-title">Drawdown</div><div class="tv-value">${m["drawdown_actuel"]:,.2f} ({m["drawdown_pct"]:.2f}%)</div></div>', unsafe_allow_html=True)

        st.markdown("### Vue consolidée de tous les comptes")
        account_totals = (
            all_trades.groupby("Compte", as_index=False)["Profit"].sum()
            if not all_trades.empty
            else pd.DataFrame(columns=["Compte", "Profit"])
        )
        accounts_meta = sheet_accounts.copy()
        if accounts_meta.empty:
            accounts_meta = pd.DataFrame({"Nom": account_names, "Initial_Capital": [10000.0] * len(account_names)})
        if "Initial_Capital" not in accounts_meta.columns:
            accounts_meta["Initial_Capital"] = 10000.0
        accounts_meta = accounts_meta.rename(columns={"Nom": "Compte"})[["Compte", "Initial_Capital"]]
        merged_accounts = pd.DataFrame({"Compte": account_names}).merge(accounts_meta, on="Compte", how="left").merge(account_totals, on="Compte", how="left")
        merged_accounts["Initial_Capital"] = pd.to_numeric(merged_accounts["Initial_Capital"], errors="coerce").fillna(10000.0)
        merged_accounts["Profit"] = pd.to_numeric(merged_accounts["Profit"], errors="coerce").fillna(0.0)
        merged_accounts["Balance ($)"] = merged_accounts["Initial_Capital"] + merged_accounts["Profit"]
        merged_accounts = merged_accounts.rename(columns={"Initial_Capital": "Capital Initial ($)", "Profit": "Profit Total ($)"})
        st.dataframe(
            merged_accounts[["Compte", "Capital Initial ($)", "Profit Total ($)", "Balance ($)"]].sort_values("Profit Total ($)", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
