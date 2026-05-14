import html
import math
import os
import time
import uuid
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

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
    discipline_score_sur_100,
    equite_series_dashboard,
    format_date_fr,
    get_trading_session,
    infer_mental_state,
    load_trades,
    filter_trades_par_compte,
    trades_papa_csv_is_effectively_empty,
    performance_par_setup_agregat,
    performance_monthly_heatmap_figure,
    load_accounts_from_sheet,
    load_ui_settings,
    resolve_trade_screenshot_path,
    save_screenshot,
    session_insights_paris_heatmap,
    strategie_conseiller_message,
    trade_timestamps_paris,
    calculer_lots,
    dash_derniers_trades_rows,
    dash_quarter_pnls,
    dash_tradevizion_score_widget,
    dash_metriques_anneaux,
    dash_pnl_par_periode,
    dash_sorties_distribution_counts,
    matsa_badges_logic,
    matsa_trade_card,
    save_trade,
    save_ui_settings,
    sync_sheet_metadata,
    upsert_account,
    generate_metric_ring_thin,
    load_trade_strategy_options,
    append_trade_strategy,
)

st.set_page_config(page_title="Mat'Sa", page_icon="📈", layout="wide")
st.session_state["data_refresh"] = True
st.session_state.df = load_trades()

if (
    "primary_font" not in st.session_state
    or "accent_color" not in st.session_state
    or "risk_per_trade_pct" not in st.session_state
):
    _ui_pref = load_ui_settings()
    st.session_state.primary_font = _ui_pref["primary_font"]
    st.session_state.accent_color = _ui_pref["accent_color"]
    try:
        _rv = float(str(_ui_pref.get("risk_per_trade_pct", "1.0")).strip().replace(",", "."))
        st.session_state.risk_per_trade_pct = max(0.1, min(5.0, _rv))
    except ValueError:
        st.session_state.risk_per_trade_pct = 1.0


def _safe_hex_color(value: str, default: str = "#00FFA3") -> str:
    s = (value or default).strip()
    if not s.startswith("#"):
        s = "#" + s
    if len(s) == 4 and s.startswith("#"):
        s = "#" + s[1] * 2 + s[2] * 2 + s[3] * 2
    if len(s) != 7:
        return default
    try:
        int(s[1:], 16)
    except ValueError:
        return default
    return s


def _hex_to_rgb_csv(h: str) -> str:
    h = _safe_hex_color(h)
    r = int(h[1:3], 16)
    g = int(h[3:5], 16)
    b = int(h[5:7], 16)
    return f"{r}, {g}, {b}"


def _compte_settings_key(account_settings: dict, name: str) -> str | None:
    """Clé canonique dans ``account_settings`` pour ``name`` (insensible à la casse)."""
    raw = str(name).strip()
    if not raw or raw == "Tous les comptes":
        return None
    if raw in account_settings:
        return raw
    kf = raw.casefold()
    for k in account_settings:
        if str(k).strip().casefold() == kf:
            return str(k)
    return None


def _compte_canonical_from_list(names: list[str], name: str) -> str | None:
    """Premier libellé de ``names`` égal à ``name`` sans tenir compte de la casse."""
    kf = str(name).strip().casefold()
    for n in names:
        if str(n).strip().casefold() == kf:
            return str(n)
    return None


def _sizing_capital_for_ui(
    selected_compte: str,
    account_settings: dict,
    account_names: list[str],
) -> float:
    ck = _compte_settings_key(account_settings, selected_compte)
    if selected_compte != "Tous les comptes" and ck is not None:
        return float(account_settings[ck].get("initial_capital", 10000.0))
    if account_names:
        return float(account_settings[account_names[0]].get("initial_capital", 10000.0))
    return float(st.session_state.get("capital_initial", 10000.0))


def _read_risk_pct_ui() -> float:
    """Curseur Paramètres (si déjà affiché) ou valeur persistée dans la session."""
    if "settings_risk_pct_slider" in st.session_state:
        return float(st.session_state["settings_risk_pct_slider"])
    return float(st.session_state.get("risk_per_trade_pct", 1.0))


def _trade_sizing_assistant_body() -> None:
    """Calculateur sizing (``st.fragment``) — hors formulaire : capital, symbole, entrée, SL, risque %, lots."""
    cap = float(st.session_state.get("_sizing_capital_cached", 10000.0))
    if "nt_trade_risk_pct" not in st.session_state:
        st.session_state.nt_trade_risk_pct = float(st.session_state.get("risk_per_trade_pct", 1.0))
    if "trade_actif" not in st.session_state:
        st.session_state.trade_actif = "NAS100"

    st.markdown('<div class="tv-card sizing-assistant-card">', unsafe_allow_html=True)
    st.markdown('<div class="tv-title">Calculateur de sizing (temps réel)</div>', unsafe_allow_html=True)
    r1c1, r1c2, r1c3 = st.columns([1.05, 1.25, 1.0])
    with r1c1:
        st.metric("Capital actuel", f"${cap:,.0f}")
    with r1c2:
        st.text_input("Symbole", key="trade_actif", placeholder="ex. NAS100, MNQ1!")
    with r1c3:
        st.number_input("Risque (%)", min_value=0.1, max_value=5.0, step=0.1, key="nt_trade_risk_pct")

    pe = float(st.session_state.get("sizing_frag_entry", 0.0))
    ps = float(st.session_state.get("sizing_frag_sl", 0.0))
    risk_pct = float(st.session_state.get("nt_trade_risk_pct", 1.0))
    inst = str(st.session_state.get("trade_actif", "NAS100")).strip() or "NAS100"
    lots, risk_usd = calculer_lots(cap, risk_pct, pe, ps, inst)
    if abs(pe - ps) < 1e-12:
        lots_safe = html.escape("N/A")
        hero_html = (
            f'<p class="nt-trade-sizing-hero">TAILLE RECOMMANDÉE : '
            f'<span class="nt-sz-val nt-sz-val--na">{lots_safe}</span> LOTS</p>'
        )
    else:
        lots_txt = f"{lots:.2f}"
        lots_safe = html.escape(lots_txt)
        sub = f"Risque estimé : ${risk_usd:,.2f} ({risk_pct:.1f} % du capital)"
        hero_html = (
            f'<p class="nt-trade-sizing-hero">TAILLE RECOMMANDÉE : '
            f'<span class="nt-sz-val nt-sz-val--num">{lots_safe}</span> LOTS</p>'
            f'<p class="sizing-risk-caption">{html.escape(sub)}</p>'
        )
    st.markdown(hero_html, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Prix d'entrée", min_value=0.0, value=0.0, step=0.01, key="sizing_frag_entry")
    with c2:
        st.number_input("Stop loss", min_value=0.0, value=0.0, step=0.01, key="sizing_frag_sl")
    st.markdown(
        "<p class=\"sizing-assistant-foot\">Les valeurs ci-dessus sont recopiées automatiquement "
        "dans le formulaire (lots & symbole) dès qu'elles changent. Tu peux les ajuster à la main dans le journal.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


_frag_deco = getattr(st, "fragment", None)
_trade_sizing_assistant_ui = _frag_deco(_trade_sizing_assistant_body) if _frag_deco else _trade_sizing_assistant_body


def matsa_sidebar_clocks() -> None:
    """Horloges NY / Paris : terminal pro, une ligne massive par ville (pleine largeur sidebar)."""
    html_sidebar_clocks = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8"/>
<style>
  html, body { margin: 0; padding: 0; overflow: hidden; background: transparent; height: 100%; width: 100%; box-sizing: border-box; }
  .matsa-clock-row {
    display: flex; justify-content: space-between; align-items: center; width: 100%; max-width: 100%;
    font-size: 1.7rem; font-weight: 800;
    font-family: Consolas, ui-monospace, "Courier New", monospace;
    box-sizing: border-box; flex-wrap: nowrap; min-height: 2.25rem;
  }
  .matsa-clock-city { color: #00FFA3; text-shadow: 0 0 8px rgba(0, 255, 163, 0.3); white-space: nowrap; }
  .matsa-clock-time { color: #FFFFFF; font-variant-numeric: tabular-nums; white-space: nowrap; }
</style>
</head>
<body>
<div class="matsa-sidebar-clocks-inner" style="margin: 0; margin-top: 20px; margin-bottom: 30px; padding: 12px 0; width: 100%; max-width: 100%; border-top: 1px solid #2A2E39; box-sizing: border-box; overflow: hidden;">
    <div style="display: flex; flex-direction: column; gap: 14px; width: 100%; margin: 0; padding: 0;">
        <div class="matsa-clock-row">
            <span class="matsa-clock-city">NEW YORK</span>
            <span id="clock-ny" class="matsa-clock-time">--:--:--</span>
        </div>
        <div class="matsa-clock-row">
            <span class="matsa-clock-city">PARIS</span>
            <span id="clock-paris" class="matsa-clock-time">--:--:--</span>
        </div>
    </div>
</div>
<script>
    function update() {
        const opt = { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
        document.getElementById('clock-ny').innerText = new Date().toLocaleTimeString('en-US', { ...opt, timeZone: 'America/New_York' });
        document.getElementById('clock-paris').innerText = new Date().toLocaleTimeString('fr-FR', { ...opt, timeZone: 'Europe/Paris' });
    }
    setInterval(update, 1000);
    update();
</script>
</body>
</html>
"""
    components.html(html_sidebar_clocks, height=148, scrolling=False, width=None)


def matsa_sidebar_upload_translate_inject() -> None:
    """Traduction FR sidebar : TreeWalker + intervalle (seule source de texte ; pas de pseudo-CSS).

    Les nœuds déjà « Charger un fichier » sont ignorés pour éviter des réécritures inutiles.
    """
    html_js = r"""
<script>
(function () {
    const forceTranslate = () => {
        try {
            const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) return;
            const walker = window.parent.document.createTreeWalker(
                sidebar,
                NodeFilter.SHOW_TEXT,
                null
            );
            let node;
            while ((node = walker.nextNode())) {
                var t = node.nodeValue;
                if (!t || !t.trim()) continue;
                var trimmed = t.trim();
                if (trimmed === 'Charger un fichier') continue;
                var nt = t;
                if (nt.indexOf('Browse files') !== -1) nt = nt.replace(/Browse files/gi, 'Parcourir…');
                if (/\bUpload\b/i.test(nt) && nt.indexOf('Charger un fichier') === -1) {
                    nt = nt.replace(/\bUpload\b/gi, 'Charger un fichier');
                }
                if (nt.indexOf('per file') !== -1) {
                    nt = nt.replace(/(\d+)\s*GB?\s*per\s*file/gi, 'Max $1 Go par fichier');
                    nt = nt.replace(/\s*per\s*file/gi, ' par fichier');
                }
                if (nt !== t) node.nodeValue = nt;
            }
        } catch (e) { /* iframe / DOM */ }
    };
    const obs = new MutationObserver(forceTranslate);
    obs.observe(window.parent.document.body, { childList: true, subtree: true });
    setInterval(forceTranslate, 500);
    forceTranslate();
})();
</script>
"""
    components.html(html_js, height=0, scrolling=False)


def _settings_panel_body() -> None:
    """Réglages Mat'Sa + feuille Accounts (ex-page Paramètres), réutilisable dans Mon Compte."""
    st.caption(
        "Gère les comptes enregistrés dans la feuille Accounts : consultation, mise à jour des limites et suppression."
    )
    st.markdown('<div class="settings-card">', unsafe_allow_html=True)
    st.markdown("#### Apparence (Mat'Sa)")
    _font_choices = ["Playfair Display", "Open Sans", "Roboto"]
    _font_ix = (
        _font_choices.index(st.session_state.primary_font)
        if st.session_state.primary_font in _font_choices
        else 0
    )
    st.selectbox(
        "Police du logo",
        _font_choices,
        index=_font_ix,
        key="primary_font",
    )
    st.color_picker(
        "Couleur d'accent (navigation, PnL positif, curseurs)",
        key="accent_color",
    )
    st.slider(
        "Risque par défaut (%)",
        min_value=0.1,
        max_value=5.0,
        value=float(st.session_state.get("risk_per_trade_pct", 1.0)),
        step=0.1,
        key="settings_risk_pct_slider",
        help="Risque cible par trade (% du capital), persisté dans matsa_ui_settings.json et utilisé par l'Assistant de Sizing.",
    )
    st.caption("Les changements visuels s'appliquent après sauvegarde et rechargement de la page.")
    st.markdown("</div>", unsafe_allow_html=True)

    accounts_df = load_accounts_from_sheet()
    st.markdown('<div class="settings-card">', unsafe_allow_html=True)
    st.markdown("#### Liste des comptes")
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
        st.dataframe(display_accounts, width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="settings-card">', unsafe_allow_html=True)
    st.markdown("#### Modifier un compte")
    if accounts_df.empty:
        st.caption("Ajoute d'abord un compte depuis la sidebar (➕ Ajouter un compte).")
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
        if st.button("Enregistrer les modifications", key="settings_save_account"):
            existing_cap = float(accounts_df[accounts_df["Nom"].astype(str) == edit_name]["Initial_Capital"].iloc[-1]) if "Initial_Capital" in accounts_df.columns else 10000.0
            upsert_account(edit_name, float(edit_obj), float(edit_loss), existing_cap)
            st.success(f"Compte '{edit_name}' mis à jour.")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="settings-card">', unsafe_allow_html=True)
    st.markdown("#### Supprimer un compte")
    if accounts_df.empty:
        st.caption("Aucun compte à supprimer.")
    else:
        names = accounts_df["Nom"].astype(str).tolist()
        del_name = st.selectbox("Compte à supprimer", names, key="settings_delete_account")
        confirm = st.checkbox(f"Confirmer la suppression définitive de '{del_name}'", key="settings_confirm_delete")
        if st.button("Supprimer ce compte", key="settings_delete_btn"):
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

    if st.button("Sauvegarder", key="settings_save_btn"):
        _rp = float(st.session_state.get("settings_risk_pct_slider", st.session_state.get("risk_per_trade_pct", 1.0)))
        _rp = max(0.1, min(5.0, _rp))
        st.session_state.risk_per_trade_pct = _rp
        save_ui_settings(
            {
                "primary_font": str(st.session_state.get("primary_font", "Playfair Display")),
                "accent_color": _safe_hex_color(str(st.session_state.get("accent_color", "#00FFA3"))),
                "risk_per_trade_pct": str(_rp),
            }
        )
        try:
            sync_sheet_metadata()
        except Exception:
            pass
        st.success("Préférences d'interface enregistrées sur disque.")
        st.rerun()


TRADE_EMOTION_RADIO_LABELS = ("Calme 🟢", "Stressé 🟠", "Euphorique 🟣", "Peur 🔴")

NT_EMOTION_SLIDER_OPTIONS = ("🧘 Calme", "😐 Modéré", "😰 Tendu", "😡 Stressé")


def _trade_emotion_canonical_from_session() -> str:
    if "nt_emotion_slider" in st.session_state:
        raw = str(st.session_state["nt_emotion_slider"])
        if "😡" in raw or "Stressé" in raw:
            return "Stressé"
        if "😰" in raw or "Tendu" in raw:
            return "Tendu"
        if "😐" in raw or "Modéré" in raw:
            return "Modéré"
        return "Calme"
    if "trade_emotion_slider" in st.session_state:
        return str(st.session_state["trade_emotion_slider"]).strip() or "Calme"
    raw = str(st.session_state.get("trade_emotion", TRADE_EMOTION_RADIO_LABELS[0]))
    for lab in TRADE_EMOTION_RADIO_LABELS:
        if raw == lab:
            return lab.split(" ", 1)[0].strip()
    return "Calme"


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


def _discipline_score(df: pd.DataFrame) -> pd.Series:
    cols = ["Sizing_Score", "SL_Score", "Revenge_Score", "Overtrading_Score", "Bias_Score"]
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=1)


def session_heatmap_paris_figure(df: pd.DataFrame, accent_hex: str = "#00FFA3") -> go.Figure:
    """Heatmap Lun–Ven × 0h–23h : somme des profits par créneau, heure Europe/Paris."""
    day_names = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi"]
    hours = list(range(24))
    base_bg = "#050505"
    accent = accent_hex if str(accent_hex).startswith("#") else "#00FFA3"

    if df.empty:
        z0 = np.zeros((5, 24))
        return go.Figure(
            data=go.Heatmap(
                z=z0,
                x=[f"{h:02d}h" for h in hours],
                y=day_names,
                showscale=False,
                hovertemplate="%{y} · %{x}<br>Aucune donnée<extra></extra>",
            )
        ).update_layout(
            title=dict(
                text="Analyse de session — sommes par jour ouvré et heure (Paris)",
                font=dict(color="#FFFFFF", size=15),
                x=0.02,
                xanchor="left",
            ),
            paper_bgcolor=base_bg,
            plot_bgcolor=base_bg,
            font={"color": "#FFFFFF"},
            margin=dict(l=12, r=28, t=52, b=12),
            height=440,
            xaxis=dict(title="Heure (Paris)", color="#A1A1AA"),
            yaxis=dict(title="", color="#A1A1AA", autorange="reversed"),
        )

    work = df.copy()
    work["Profit"] = pd.to_numeric(work["Profit"], errors="coerce").fillna(0.0)
    work["_ts"] = trade_timestamps_paris(work["Date"])
    work = work.dropna(subset=["_ts"])
    work["_dow"] = work["_ts"].dt.dayofweek
    work = work[work["_dow"] <= 4]
    work["_hour"] = work["_ts"].dt.hour.clip(0, 23)
    work["_day"] = work["_dow"].map(lambda i: day_names[i] if i < 5 else None)
    work = work.dropna(subset=["_day"])
    if work.empty:
        z0 = np.zeros((5, 24))
        return go.Figure(
            data=go.Heatmap(
                z=z0,
                x=[f"{h:02d}h" for h in hours],
                y=day_names,
                showscale=False,
                hovertemplate="%{y} · %{x}<br>Aucune donnée<extra></extra>",
            )
        ).update_layout(
            title=dict(
                text="Analyse de session — sommes par jour ouvré et heure (Paris)",
                font=dict(color="#FFFFFF", size=15),
                x=0.02,
                xanchor="left",
            ),
            paper_bgcolor=base_bg,
            plot_bgcolor=base_bg,
            font={"color": "#FFFFFF"},
            margin=dict(l=12, r=28, t=52, b=12),
            height=440,
            xaxis=dict(title="Heure (Paris)", color="#A1A1AA"),
            yaxis=dict(title="", color="#A1A1AA", autorange="reversed"),
        )

    agg = work.groupby(["_day", "_hour"], as_index=False)["Profit"].sum()
    pivot = agg.pivot(index="_day", columns="_hour", values="Profit").reindex(index=day_names)
    for h in hours:
        if h not in pivot.columns:
            pivot[h] = np.nan
    pivot = pivot.reindex(columns=hours)
    z = np.nan_to_num(pivot.values.astype(float), nan=0.0)
    zmax_abs = float(np.nanmax(np.abs(z))) if z.size else 0.0
    if zmax_abs < 1e-9:
        zmax_abs = 1e-9
    zmin, zmax = -zmax_abs, zmax_abs

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[f"{h:02d}h" for h in hours],
            y=day_names,
            zmin=zmin,
            zmax=zmax,
            zmid=0.0,
            colorscale=[
                [0.0, "#B91C1C"],
                [0.35, "#450A0A"],
                [0.5, "#050505"],
                [0.65, "#052e22"],
                [1.0, accent],
            ],
            hovertemplate="%{y} · %{x} (Paris)<br>Somme des profits : %{z:,.2f} $<extra></extra>",
            colorbar=dict(
                title=dict(text="Profit ($)", side="right", font=dict(color="#E5E7EB")),
                tickfont=dict(color="#A1A1AA"),
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text="Analyse de session — sommes par jour ouvré et heure (Paris)",
            font=dict(color="#FFFFFF", size=15),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor=base_bg,
        plot_bgcolor=base_bg,
        font={"color": "#FFFFFF"},
        margin=dict(l=12, r=28, t=52, b=12),
        height=440,
        xaxis=dict(title="Heure (Paris)", color="#A1A1AA", gridcolor="#1F2937", showgrid=False),
        yaxis=dict(title="", color="#A1A1AA", autorange="reversed"),
        hoverlabel={"bgcolor": "#13161D", "font": {"color": "#FFFFFF"}},
    )
    return fig


def discipline_progress_ring_html(score_pct: float, accent_hex: str, accent_rgb_csv: str) -> str:
    """Anneau de progression HTML/SVG (0–100) avec lueur couleur d'accent."""
    pct = max(0.0, min(100.0, float(score_pct)))
    r = 40.0
    circ = 2.0 * math.pi * r
    dash_off = circ * (1.0 - pct / 100.0)
    ah = accent_hex if str(accent_hex).startswith("#") and len(str(accent_hex)) >= 7 else "#00FFA3"
    ah = html.escape(ah[:7])
    label = f"{pct:.0f}"
    return f"""
    <div class="matsa-disc-wrap" style="display:flex;flex-direction:column;align-items:center;margin:0.25rem 0 1.35rem 0;">
        <div class="matsa-disc-ring-host" style="position:relative;width:168px;height:168px;">
            <svg width="168" height="168" viewBox="0 0 100 100" aria-hidden="true"
                style="transform:rotate(-90deg);filter:drop-shadow(0 0 10px rgba({accent_rgb_csv},0.55)) drop-shadow(0 0 22px rgba({accent_rgb_csv},0.2));">
                <circle cx="50" cy="50" r="{r:.1f}" fill="none" stroke="#27272A" stroke-width="4.2"/>
                <circle cx="50" cy="50" r="{r:.1f}" fill="none" stroke="{ah}" stroke-width="4.2"
                    stroke-linecap="round" stroke-dasharray="{circ:.4f}" stroke-dashoffset="{dash_off:.4f}"/>
            </svg>
            <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;pointer-events:none;">
                <span style="font-size:2.05rem;font-weight:800;color:#F9FAFB;letter-spacing:-0.02em;">{label}</span>
            </div>
        </div>
        <p style="color:#9CA3AF;font-size:0.88rem;margin:0.15rem 0 0 0;text-align:center;max-width:22rem;line-height:1.35;">
            Discipline globale de la période
        </p>
    </div>
    """


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


def _accent_fill_rgba(accent_hex: str, alpha: float) -> str:
    h = (accent_hex or "#00FFA3").strip().lstrip("#")
    if len(h) == 6:
        try:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
        except ValueError:
            pass
    return f"rgba(0,255,163,{alpha})"


def equity_curve_figure(
    df: pd.DataFrame,
    accent_hex: str,
    *,
    height: int = 400,
    paper_bg: str | None = None,
) -> go.Figure:
    """Area chart PnL cumulé : remplissage dégradé vert néon → bleu, ligne d'accent."""
    d = equite_series_dashboard(df)
    base = paper_bg or "#0b0e14"
    accent = accent_hex if str(accent_hex).startswith("#") and len(str(accent_hex)) >= 7 else "#00FFA3"
    if not d["dates"]:
        return go.Figure().update_layout(
            paper_bgcolor=base,
            plot_bgcolor=base,
            font=dict(color="#E5E7EB"),
            height=max(260, int(height)),
            title=dict(
                text="Courbe d'équité — ajoute des trades pour afficher la performance",
                font=dict(size=14, color="#9CA3AF"),
            ),
            margin=dict(l=16, r=16, t=48, b=16),
        )
    x = d["dates"]
    y_eq = d["equity"]
    labels = d.get("date_labels") or []
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_eq,
            mode="lines",
            line=dict(color=accent, width=3.2),
            fill="tozeroy",
            fillgradient=dict(
                type="vertical",
                colorscale=[
                    [0.0, "rgba(0, 255, 200, 0.50)"],
                    [0.45, "rgba(0, 200, 230, 0.22)"],
                    [1.0, "rgba(30, 80, 180, 0.04)"],
                ],
            ),
            fillcolor="rgba(0,0,0,0)",
            name="PnL cumulé",
            customdata=labels if len(labels) == len(x) else None,
            hovertemplate=(
                "%{customdata}<br>PnL cumulé : %{y:,.2f} $<extra></extra>"
                if labels and len(labels) == len(x)
                else "%{x}<br>PnL cumulé : %{y:,.2f} $<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text="Performance cumulée (area)",
            font=dict(color="#F9FAFB", size=15),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor=base,
        plot_bgcolor=base,
        font=dict(color="#E5E7EB"),
        margin=dict(l=16, r=16, t=52, b=44),
        height=int(height),
        xaxis=dict(
            title="Date",
            color="#9CA3AF",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="PnL cumulé ($)",
            color="#9CA3AF",
            showgrid=False,
            zeroline=True,
            zerolinecolor="#27272A",
        ),
        hoverlabel=dict(bgcolor="#111318", font=dict(color="#FFFFFF")),
        showlegend=False,
    )
    return fig


def _dashboard_kpi_mini_html(label: str, value: float, delta: float | None = None) -> str:
    """Mini-carte KPI (libellé gris au-dessus, montant $ vert/rouge, delta gris optionnel)."""
    lbl = html.escape(label.upper())
    v_s = html.escape(f"${value:,.2f}")
    val_cls = (
        "tv-card-dash-finance__value tv-card-dash-finance__value--profit tv-dash-kpi-mini__value"
        if value >= 0
        else "tv-card-dash-finance__value tv-card-dash-finance__value--loss tv-dash-kpi-mini__value"
    )
    delta_html = ""
    if delta is not None:
        d_sign = "+" if float(delta) >= 0 else ""
        d_s = html.escape(f"Δ {d_sign}${float(delta):,.2f}")
        delta_html = f'<div class="tv-dash-kpi-mini__delta">{d_s}</div>'
    return (
        f'<div class="matsa-dash-card tv-dash-kpi-mini">'
        f'<div class="tv-dash-kpi-mini__label">{lbl}</div>'
        f'<div class="{val_cls}">{v_s}</div>'
        f"{delta_html}"
        "</div>"
    )


def _dashboard_badges_html(badges: list[dict[str, str]]) -> str:
    """Petit bloc icône + texte sous la jauge de discipline."""
    if not badges:
        return (
            '<div class="tv-dash-badges tv-dash-badges--empty matsa-dash-card matsa-dash-card--badges">'
            '<span class="tv-dash-badges__muted">Aucun badge actif</span></div>'
        )
    parts = ['<div class="tv-dash-badges matsa-dash-card matsa-dash-card--badges">']
    for b in badges:
        ic = html.escape(str(b.get("icon", "")))
        lb = html.escape(str(b.get("label", "")))
        parts.append(
            f'<span class="tv-dash-badge">{ic}<span class="tv-dash-badge__txt">{lb}</span></span>'
        )
    parts.append("</div>")
    return "".join(parts)


def _dashboard_exit_distribution_html(counts: dict[str, int]) -> str:
    """Barres horizontales Markdown/CSS pour TP / TP Partiel / BE / SL."""
    order = (
        ("TP", "#00FFA3"),
        ("TP Partiel", "#EAB308"),
        ("BE", "#00D1FF"),
        ("SL", "#FF4B4B"),
    )
    total = max(1, sum(int(counts.get(k, 0)) for k, _ in order))
    labels_map = {"TP": "Full TP", "TP Partiel": "Partiels", "BE": "Breakeven", "SL": "SL Hit"}
    parts: list[str] = [
        '<div class="tv-dash-exit-wrap matsa-dash-card matsa-dash-card--pad tvz-dist-wrap">',
        '<div class="tvz-dist-head">distribution</div>',
    ]
    for name, color in order:
        n = int(counts.get(name, 0))
        pct = 100.0 * n / total
        ne = html.escape(str(n))
        name_e = html.escape(labels_map.get(name, name))
        parts.append(
            f'<div class="tv-dash-exit-row">'
            f'<span class="tv-dash-exit-row__name">{name_e}</span>'
            f'<div class="tv-dash-exit-row__track">'
            f'<div class="tv-dash-exit-row__fill" style="width:{pct:.1f}%;background:{color};"></div>'
            "</div>"
            f'<span class="tv-dash-exit-row__n">{ne}</span>'
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _dashboard_highlight_card(emoji: str, label: str, value_text: str, val_extra_class: str = "") -> str:
    ec = val_extra_class.strip()
    if ec not in ("tv-dash-highlight__val--up", "tv-dash-highlight__val--down"):
        ec = ""
    extra = f" {ec}" if ec else ""
    return (
        f'<div class="matsa-dash-card tv-dash-highlight">'
        f'<div class="tv-dash-highlight__emoji">{html.escape(emoji)}</div>'
        f'<div class="tv-dash-highlight__lbl">{html.escape(label)}</div>'
        f'<div class="tv-dash-highlight__val{extra}">{html.escape(value_text)}</div>'
        "</div>"
    )


def _dashboard_glory_line_html(n_comptes: int, profit_global: float, best_streak: int) -> str:
    """Barre résumé tout en haut du Dashboard (Ligne de Gloire)."""
    nc = max(0, int(n_comptes))
    zs = max(0, int(best_streak))
    if profit_global >= 0:
        profit_s = html.escape(f"+${profit_global:,.2f}")
        p_cls = "matsa-dash-glory__profit matsa-dash-glory__profit--up"
    else:
        profit_s = html.escape(f"-${abs(profit_global):,.2f}")
        p_cls = "matsa-dash-glory__profit matsa-dash-glory__profit--down"
    return (
        f'<div class="matsa-dash-glory" role="note">'
        f'<div class="matsa-dash-glory__body">'
        f"<span>Total comptes : <strong>{nc}</strong></span>"
        f'<span class="matsa-dash-glory__sep">|</span>'
        f'<span>Profit global : <strong class="{p_cls}">{profit_s}</strong></span>'
        f'<span class="matsa-dash-glory__sep">|</span>'
        f"<span>Meilleure série : <strong>{zs}</strong> trades</span>"
        f"</div>"
        f'<div class="matsa-dash-glory__flow" aria-hidden="true"></div>'
        f"</div>"
    )


def _dashboard_max_daily_loss_usd(
    selected_compte: str,
    account_settings: dict[str, dict[str, float]],
    account_names: list[str],
) -> float:
    """Seuil Max Daily Loss ($) pour le compte filtré, ou le plus strict si « Tous les comptes »."""
    ck = _compte_settings_key(account_settings, selected_compte)
    if selected_compte != "Tous les comptes" and ck is not None:
        return float(account_settings[ck].get("max_daily_loss_usd", 500.0))
    if not account_names:
        return 500.0
    return min(
        float(account_settings.get(n, {}).get("max_daily_loss_usd", 500.0)) for n in account_names
    )


def _dashboard_insight_gemini_html(message: str) -> str:
    """Bulle insight type IA (glass + étincelle)."""
    msg = html.escape((message or "").strip())
    return (
        f'<div class="matsa-dash-card matsa-dash-insight" role="status">'
        f'<div class="matsa-dash-insight__icon" aria-hidden="true">✨</div>'
        f'<div class="matsa-dash-insight__txt">{msg}</div>'
        f"</div>"
    )


def _tvz_money_signed(v: float) -> str:
    if float(v) >= 0:
        return f"+${float(v):,.2f}"
    return f"-${abs(float(v)):,.2f}"


def _tvz_header_card_html(
    title: str,
    value_inner_html: str,
    value_class: str,
    subline_inner_html: str,
    accent: str,
) -> str:
    """Carte en-tête TradeVizion (bordure supérieure 3px colorée)."""
    acc = accent.strip().lower()
    if acc not in ("green", "purple", "cyan", "blue", "yellow", "orange"):
        acc = "green"
    return (
        f'<div class="tvz-hdr-card tvz-hdr-card--{html.escape(acc)}">'
        f'<div class="tvz-hdr-card__label">{html.escape(title.upper())}</div>'
        f'<div class="tvz-hdr-card__value {html.escape(value_class)}">{value_inner_html}</div>'
        f'<div class="tvz-hdr-card__sub">{subline_inner_html}</div>'
        "</div>"
    )


def _tvz_frac_class(a: int, b: int) -> str:
    if b <= 0:
        return "tvz-frac--mid"
    r = float(a) / float(b)
    if r >= 0.66:
        return "tvz-frac--good"
    if r >= 0.33:
        return "tvz-frac--mid"
    return "tvz-frac--bad"


def _tvz_score_center_html(d: dict) -> str:
    """Bloc score central (Mat'Sa / logique ``dash_tradevizion_score_widget``)."""
    main = int(d["main"])
    lab = html.escape(str(d["label"]))
    hint_raw = str(d.get("hint") or "").strip()
    hint = html.escape(hint_raw) if hint_raw else ""
    ps = int(d["perf_side"])
    pm = int(d["psycho_side"])
    pairs = (
        ("Volume", d["volume"]),
        ("Prof. Ratio", d["prof_ratio"]),
        ("Consistance", d["consistance"]),
        ("Risque", d["risk"]),
        ("Qualité", d["qual"]),
        ("Plan & Règles", d["plan"]),
    )
    cells = []
    for lk, (a, b) in pairs:
        ia, ib = int(a), int(b)
        fc = _tvz_frac_class(ia, ib)
        cells.append(
            f'<div class="tvz-score-cell">'
            f'<span class="tvz-score-cell__k">{html.escape(lk)}</span>'
            f'<span class="tvz-score-cell__v {fc}">{ia}/{ib}</span>'
            "</div>"
        )
    grid = f'<div class="tvz-score-grid">{"".join(cells)}</div>'
    hint_html = f'<div class="tvz-score-hint">{hint}</div>' if hint else ""
    return (
        f'<div class="matsa-dash-card tvz-score-card">'
        f'<div class="tvz-score-mid">'
        f'<div class="tvz-score-side tvz-score-side--perf">'
        f'<div class="tvz-score-side__lbl">PERF</div>'
        f'<div class="tvz-score-side__score">'
        f'<span class="tvz-score-side__nv">{ps}</span>'
        f'<span class="tvz-score-side__sf">/100</span></div></div>'
        f'<div class="tvz-score-center">'
        f'<div class="tvz-score-big">{main}<span class="tvz-score-suf">/100</span></div>'
        f'<div class="tvz-score-lbl">{lab}</div>'
        f"{hint_html}"
        f"</div>"
        f'<div class="tvz-score-side tvz-score-side--psycho">'
        f'<div class="tvz-score-side__lbl">PSYCHO</div>'
        f'<div class="tvz-score-side__score">'
        f'<span class="tvz-score-side__nv">{pm}</span>'
        f'<span class="tvz-score-side__sf">/100</span></div></div>'
        f"</div>"
        f"{grid}"
        "</div>"
    )


def _tvz_section_title(text: str) -> str:
    t = str(text).strip().upper()
    return f'<div class="tvz-section-title">{html.escape(t)}</div>'


def _tvz_ring_dash_placeholder(label: str) -> str:
    """Anneau remplacé par un tiret (PF / R:R non définis sans pertes)."""
    le = html.escape(str(label).strip().upper())
    return (
        f'<div class="mastery-ring-thin-wrap matsa-dash-card matsa-dash-card--ring tvz-gauge-compact tvz-ring-dash">'
        f'<div class="tvz-gauge-svg-wrap tvz-gauge-svg-wrap--dash">'
        f'<div class="tvz-ring-dash__circle" aria-hidden="true">—</div></div>'
        f'<div class="tvz-gauge-val">—</div>'
        f'<div class="mastery-ring-thin-label tvz-gauge-lbl">{le}</div>'
        "</div>"
    )


def underwater_drawdown_figure(df: pd.DataFrame) -> go.Figure:
    """Drawdown normalisé : (équité − sommet historique) / sommet ; zone rouge jusqu’à 0 %."""
    base = "#050505"
    d = equite_series_dashboard(df)
    if not d["dates"]:
        return go.Figure().update_layout(
            paper_bgcolor=base,
            plot_bgcolor=base,
            font=dict(color="#E5E7EB"),
            height=360,
            title=dict(
                text="Underwater drawdown — pas assez de données",
                font=dict(size=14, color="#9CA3AF"),
            ),
            margin=dict(l=16, r=16, t=48, b=16),
        )
    eq = np.asarray(d["equity"], dtype=float)
    peak = np.asarray(d["peak"], dtype=float)
    dd = np.zeros_like(eq, dtype=float)
    mask = peak > 1e-9
    dd[mask] = (eq[mask] - peak[mask]) / peak[mask]
    dd_pct = dd * 100.0
    x = d["dates"]
    labels = d.get("date_labels") or []
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=dd_pct,
            mode="lines",
            line=dict(color="#EF4444", width=2.2),
            name="Drawdown",
            fill="tozeroy",
            fillcolor="rgba(220, 38, 38, 0.45)",
            customdata=labels if len(labels) == len(x) else None,
            hovertemplate=(
                "%{customdata}<br>Drawdown : %{y:.2f} %<extra></extra>"
                if labels and len(labels) == len(x)
                else "%{x}<br>Drawdown : %{y:.2f} %<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text="Underwater drawdown (PnL cumulé vs sommet)",
            font=dict(color="#F9FAFB", size=14),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor=base,
        plot_bgcolor=base,
        font=dict(color="#E5E7EB"),
        margin=dict(l=16, r=16, t=56, b=44),
        height=380,
        xaxis=dict(title="Date", color="#9CA3AF", showgrid=False),
        yaxis=dict(
            title="(Equité − Sommet) / Sommet (%)",
            color="#9CA3AF",
            showgrid=True,
            gridcolor="#1a1a1f",
            zeroline=True,
            zerolinecolor="#4B5563",
        ),
        hoverlabel=dict(bgcolor="#111318", font=dict(color="#FFFFFF")),
        showlegend=False,
    )
    return fig


def setup_pnl_bar_figure(stats: pd.DataFrame, accent_hex: str) -> go.Figure:
    """Barres : PnL cumulé par stratégie (somme des profits), accent si positif, gris acier si négatif."""
    base = "#050505"
    steel = "#8B98A8"
    accent = _safe_hex_color(accent_hex)
    if stats.empty:
        return go.Figure().update_layout(
            paper_bgcolor=base,
            plot_bgcolor=base,
            font=dict(color="#E5E7EB"),
            height=320,
            title=dict(text="Aucune donnée par stratégie.", font=dict(size=14, color="#9CA3AF")),
            margin=dict(l=16, r=16, t=48, b=16),
        )
    x = stats["Setup"].astype(str).tolist()
    y = [float(v) for v in stats["Profit_Net"].tolist()]
    colors = [accent if v >= 0 else steel for v in y]
    fig = go.Figure(
        go.Bar(
            x=x,
            y=y,
            marker_color=colors,
            marker_line_width=0,
            hovertemplate="%{x}<br>PnL cumulé : %{y:,.2f} $<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text="PnL cumulé par stratégie",
            font=dict(color="#F9FAFB", size=15),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor=base,
        plot_bgcolor=base,
        font=dict(color="#E5E7EB"),
        margin=dict(l=16, r=16, t=52, b=88),
        height=400,
        xaxis=dict(title="Setup", color="#9CA3AF", tickangle=-24),
        yaxis=dict(
            title="PnL cumulé ($)",
            color="#9CA3AF",
            showgrid=True,
            gridcolor="#1a1a1f",
            zeroline=True,
            zerolinecolor="#27272A",
        ),
        hoverlabel=dict(bgcolor="#111318", font=dict(color="#FFFFFF")),
        showlegend=False,
    )
    return fig


def trader_brain_radar_figure(df: pd.DataFrame, accent_hex: str = "#00FFA3") -> go.Figure:
    """Radar : moyenne des 5 critères d'exécution (profil psychologique / discipline)."""
    pairs = [
        ("Sizing", "Sizing_Score"),
        ("Over-trading", "Overtrading_Score"),
        ("Gestion SL", "SL_Score"),
        ("Cohérence biais", "Bias_Score"),
        ("Contrôle Revenge", "Revenge_Score"),
    ]
    labels = [p[0] for p in pairs]
    cols = [p[1] for p in pairs]
    sub = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    means = [float(sub[c].mean()) for c in cols]
    means = [max(0.0, min(20.0, m)) for m in means]
    theta = labels + [labels[0]]
    r = means + [means[0]]
    fill_col = _accent_fill_rgba(accent_hex, 0.2)
    line_col = accent_hex if accent_hex.startswith("#") else "#00FFA3"

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=r,
                theta=theta,
                fill="toself",
                fillcolor=fill_col,
                line=dict(color=line_col, width=2.2),
                mode="lines+markers",
                marker=dict(size=7, color=line_col, line=dict(width=0, color=line_col)),
                name="Moyenne",
                hovertemplate="%{theta}<br>Score moyen : %{r:.1f} / 20<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(
            text="Équilibre des critères d'exécution (moyenne 0–20)",
            font=dict(color="#FFFFFF", size=15),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"),
        margin=dict(l=48, r=48, t=52, b=48),
        height=460,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 20],
                tickvals=[0, 5, 10, 15, 20],
                tickfont=dict(color="#9CA3AF", size=11),
                gridcolor="rgba(55, 65, 81, 0.35)",
                linecolor="rgba(55, 65, 81, 0.45)",
                showline=True,
            ),
            angularaxis=dict(
                tickfont=dict(color="#E5E7EB", size=12),
                gridcolor="rgba(55, 65, 81, 0.28)",
                linecolor="rgba(55, 65, 81, 0.4)",
            ),
        ),
        showlegend=False,
        hoverlabel=dict(bgcolor="#13161D", font=dict(color="#FFFFFF")),
    )
    return fig


@st.dialog("Graphique du trade", width="large", on_dismiss="ignore")
def trade_chart_dialog(resolved_path: str) -> None:
    """Aperçu plein écran du graphique ; fermeture via le bouton ou la croix native."""
    _, c_img, _ = st.columns([0.08, 0.84, 0.08])
    with c_img:
        if resolved_path:
            try:
                st.image(resolved_path, width="stretch")
            except Exception:
                st.markdown(
                    '<p style="color:#9CA3AF;text-align:center;margin:2rem 0;">Aucun visuel disponible</p>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<p style="color:#9CA3AF;text-align:center;margin:2rem 0;">Aucun visuel disponible</p>',
                unsafe_allow_html=True,
            )
    _, c_btn, _ = st.columns([0.42, 0.16, 0.42])
    with c_btn:
        if st.button("Fermer", key="trade_chart_dialog_close", type="secondary"):
            st.rerun()


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


if "_matsa_sheet_init_done" not in st.session_state:
    try:
        sync_sheet_metadata()
    except Exception:
        pass
    st.session_state._matsa_sheet_init_done = True

_accent = _safe_hex_color(str(st.session_state.get("accent_color", "#00FFA3")))
_font_logo = str(st.session_state.get("primary_font", "Playfair Display")).replace('"', "").replace(";", "")
_accent_rgb = _hex_to_rgb_csv(_accent)

st.markdown(
    f"""
    <style>
        @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Open+Sans:ital,wght@0,400;0,700;1,400&family=Playfair+Display:ital,wght@0,900;1,900&family=Roboto:ital,wght@0,400;0,700;1,400&display=swap");
        @import url("https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0");

        :root {{
            --matsa-accent: {_accent};
            --matsa-accent-rgb: {_accent_rgb};
            --matsa-font-logo: "{_font_logo}", Georgia, "Times New Roman", serif;
        }}

        /* Évite le défilement horizontal (ex. marges négatives pleine largeur) */
        html, body,
        [data-testid="stAppViewContainer"],
        .stApp {{
            overflow-x: hidden !important;
            max-width: 100vw !important;
        }}

        summary::-webkit-details-marker {{ display: none !important; }}

        @-webkit-keyframes matsaLogoShimmer {{
            0% {{ background-position: -120% 0, 0% 45%, 0% 50%; }}
            100% {{ background-position: 220% 0, 0% 55%, 0% 50%; }}
        }}
        @keyframes matsaLogoShimmer {{
            0% {{ background-position: -120% 0, 0% 45%, 0% 50%; }}
            100% {{ background-position: 220% 0, 0% 55%, 0% 50%; }}
        }}

        /* Logo Mat'Sa — or + balayage shimmer (3–4 s) */
        .matsa-logo {{
            display: block !important;
            text-align: center;
            font-size: 52px !important;
            font-style: italic !important;
            font-family: var(--matsa-font-logo) !important;
            font-weight: 900 !important;
            letter-spacing: 0.02em;
            line-height: 0.95;
            margin: -20px auto 30px auto !important;
            position: relative;
            background:
                linear-gradient(
                    90deg,
                    transparent 0%,
                    #BF953F 18%,
                    #FCF6BA 42%,
                    #BF953F 62%,
                    transparent 82%,
                    transparent 100%
                ),
                radial-gradient(ellipse 120% 80% at 50% 0%, rgba(255,255,255,0.48) 0%, transparent 42%),
                linear-gradient(185deg,
                    #fffef2 0%,
                    #ffd700 8%,
                    #c9a227 22%,
                    #fff8dc 36%,
                    #8b6914 50%,
                    #f5e6a8 64%,
                    #b8860b 78%,
                    #ffe566 88%,
                    #daa520 100%) !important;
            background-size: 280% 100%, 200% 200%, 200% 200% !important;
            background-repeat: no-repeat !important;
            background-position: -80% 0, 0% 50%, 0% 50% !important;
            -webkit-animation: matsaLogoShimmer 3.6s ease-in-out infinite !important;
            animation: matsaLogoShimmer 3.6s ease-in-out infinite !important;
            -webkit-background-clip: text !important;
            background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            color: transparent !important;
            filter: drop-shadow(0 1px 0 rgba(0,0,0,0.45)) drop-shadow(0 0 12px rgba(255, 215, 0, 0.22));
            will-change: background-position;
            transform: translateZ(0);
            backface-visibility: hidden;
        }}
        .matsa-logo::before, .matsa-logo::after {{
            content: '✦';
            position: absolute;
            font-size: 16px;
            font-style: normal !important;
            color: #ffd700;
            text-shadow: 0 0 8px rgba(255, 215, 0, 0.55);
            -webkit-text-fill-color: #ffd700;
            pointer-events: none;
        }}
        .matsa-logo::before {{ top: 0; right: 10%; }}
        .matsa-logo::after {{ bottom: 0; left: 10%; }}

        /* Sidebar — fond */
        [data-testid="stSidebar"] {{ background-color: #0E1117 !important; border-right: 1px solid #1f2937; }}
        [data-testid="stSidebarContent"] {{ padding-top: 0px !important; }}

        /* Sidebar — navigation TradeVizion (boutons natifs, plus de st.radio) */
        .tvz-nav-intro {{
            color: #6B7280;
            font-size: 0.62rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 6px 0 8px 2px;
        }}
        [data-testid="stSidebar"] [class*="st-key-tvznav_"] button[kind="secondary"] {{
            width: 100% !important;
            justify-content: flex-start !important;
            align-items: center !important;
            text-align: left !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-left: 4px solid transparent !important;
            background: rgba(22, 26, 37, 0.92) !important;
            color: #E4E7EC !important;
            font-weight: 600 !important;
            font-size: 0.86rem !important;
            padding: 0.52rem 0.75rem 0.52rem 0.65rem !important;
            min-height: 0 !important;
            box-shadow: none !important;
        }}
        [data-testid="stSidebar"] [class*="st-key-tvznav_"] button[kind="primary"] {{
            width: 100% !important;
            justify-content: flex-start !important;
            align-items: center !important;
            text-align: left !important;
            border-radius: 10px !important;
            border: 1px solid rgba(99, 102, 241, 0.45) !important;
            border-left: 5px solid #38BDF8 !important;
            background: linear-gradient(
                92deg,
                rgba(99, 102, 241, 0.42) 0%,
                rgba(59, 130, 246, 0.22) 45%,
                rgba(30, 27, 75, 0.35) 100%
            ) !important;
            box-shadow:
                0 0 24px rgba(56, 189, 248, 0.28),
                inset 0 1px 0 rgba(255, 255, 255, 0.07) !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            font-size: 0.86rem !important;
            padding: 0.52rem 0.75rem 0.52rem 0.65rem !important;
        }}
        [data-testid="stSidebar"] [class*="st-key-tvznav_"] button[kind="secondary"]:hover,
        [data-testid="stSidebar"] [class*="st-key-tvznav_"] button[kind="primary"]:hover {{
            border-color: rgba(148, 163, 184, 0.35) !important;
            color: #FFFFFF !important;
        }}
        [data-testid="stSidebar"] [class*="st-key-tvznav_"] button[kind="primary"]::before,
        [data-testid="stSidebar"] [class*="st-key-tvznav_"] button[kind="secondary"]::before {{
            font-family: "Material Symbols Outlined" !important;
            font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24 !important;
            font-weight: normal !important;
            font-style: normal !important;
            font-size: 1.28rem !important;
            line-height: 1 !important;
            margin-right: 0.55rem !important;
            vertical-align: -0.18em !important;
            display: inline-block !important;
            -webkit-font-smoothing: antialiased !important;
            font-feature-settings: "liga" !important;
            color: inherit !important;
        }}
        [data-testid="stSidebar"] .st-key-tvznav_dash button::before {{ content: "grid_view" !important; }}
        [data-testid="stSidebar"] .st-key-tvznav_suivi button::before {{ content: "calendar_today" !important; }}
        [data-testid="stSidebar"] .st-key-tvznav_news button::before {{ content: "newspaper" !important; }}
        [data-testid="stSidebar"] .st-key-tvznav_stats button::before {{ content: "bar_chart" !important; }}
        [data-testid="stSidebar"] .st-key-tvznav_analyses button::before {{ content: "analytics" !important; }}
        [data-testid="stSidebar"] .st-key-tvznav_trading button::before {{ content: "menu_book" !important; }}
        [data-testid="stSidebar"] .st-key-tvznav_compte button::before {{ content: "account_balance_wallet" !important; }}
        [data-testid="stSidebar"] .st-key-tvznav_nouveau button::before {{ content: "add_circle" !important; }}

        .matsa-dash-rings-gap {{
            height: 18px !important;
            margin: 0 !important;
            padding: 0 !important;
            pointer-events: none !important;
        }}

        /* Mastery Rings — une ligne dense (mobile / desktop) */
        .mastery-ring-wrap {{
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: flex-start !important;
            width: 100% !important;
            min-width: 0 !important;
            text-align: center !important;
        }}
        .mastery-ring-wrap svg.mastery-ring-svg {{
            display: block !important;
            margin: 0 auto !important;
        }}
        .mastery-ring-label {{
            color: #848E9C !important;
            font-size: 0.62rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
            margin-top: 2px !important;
            width: 100% !important;
            max-width: 100% !important;
            text-align: center !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }}
        [data-testid="stHorizontalBlock"]:has(.mastery-ring-wrap) {{
            display: flex !important;
            justify-content: space-between !important;
            align-items: flex-start !important;
            gap: 0.5rem !important;
            width: 100% !important;
            flex-wrap: nowrap !important;
        }}
        [data-testid="stHorizontalBlock"]:has(.mastery-ring-wrap) > div[data-testid="column"] {{
            flex: 1 1 0 !important;
            min-width: 0 !important;
            padding-left: 2px !important;
            padding-right: 2px !important;
            display: flex !important;
            justify-content: center !important;
        }}
        [data-testid="stHorizontalBlock"]:has(.mastery-ring-wrap) > div[data-testid="column"] > div {{
            width: 100% !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }}

        /* Émotion — 4 choix type boutons, bordure néon à la sélection */
        div.st-key-trade_emotion div[role="radiogroup"] {{
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: wrap !important;
            gap: 10px !important;
            width: 100% !important;
        }}
        div.st-key-trade_emotion div[role="radiogroup"] label {{
            flex: 1 1 18% !important;
            min-width: 118px !important;
            justify-content: center !important;
            border-radius: 12px !important;
            border: 2px solid rgba(255, 255, 255, 0.14) !important;
            background: rgba(15, 17, 23, 0.9) !important;
            padding: 12px 8px !important;
            margin: 0 !important;
            cursor: pointer !important;
        }}
        div.st-key-trade_emotion div[role="radiogroup"] label[data-checked="true"] {{
            border-color: var(--matsa-accent) !important;
            box-shadow:
                0 0 18px rgba(var(--matsa-accent-rgb), 0.55),
                inset 0 0 14px rgba(var(--matsa-accent-rgb), 0.12) !important;
            background: rgba(255, 255, 255, 0.06) !important;
        }}
        div.st-key-trade_emotion div[role="radiogroup"] input[type="radio"],
        div.st-key-trade_emotion div[role="radiogroup"] label svg,
        div.st-key-trade_emotion div[role="radiogroup"] label [data-baseweb="radio"] {{
            display: none !important;
            width: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        div.st-key-trade_emotion div[role="radiogroup"] label > input + div {{
            display: flex !important;
            flex: 1 1 auto !important;
            justify-content: center !important;
            align-items: center !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        div.st-key-trade_emotion label p {{
            font-size: 0.98rem !important;
            font-weight: 700 !important;
            text-align: center !important;
        }}

        .matsa-sidebar-import-head {{
            margin-top: 30px !important;
            margin-bottom: 10px !important;
            font-size: 0.74rem !important;
            font-weight: 400 !important;
            letter-spacing: 0.04em !important;
            color: #6B7280 !important;
            line-height: 1.45 !important;
        }}

        /* Espace minimal sous les horloges (le bloc horloge a déjà margin-bottom) */
        .matsa-sidebar-clocks-spacer {{
            display: block !important;
            height: 0 !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] {{
            margin-top: 0 !important;
            padding-top: 0.15rem !important;
        }}

        /* Sidebar — zone d'upload : bordure, fond semi-transparent, hover néon */
        [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {{
            border: 1px dashed #2A2E39 !important;
            border-radius: 8px !important;
            background: rgba(22, 26, 37, 0.5) !important;
            transition: border-color 0.22s ease, box-shadow 0.22s ease, background 0.22s ease !important;
        }}
        [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"]:hover {{
            border-color: #00FFA3 !important;
            box-shadow: 0 0 0 1px rgba(0, 255, 163, 0.25) !important;
        }}

        /* Uploader — méthode nucléaire : on neutralise TOUS les textes/icônes natifs
           via font-size:0 + suppression des SVG, puis on rétablit uniquement
           le bouton et la balise <small> qu'on veut afficher. */
        [data-testid="stFileUploadDropzone"] {{
            font-size: 0 !important; /* Cache tous les textes natifs ("File", "Drag and drop file here", etc.) */
            color: transparent !important;
        }}
        [data-testid="stFileUploadDropzone"] svg,
        [data-testid="stFileUploadDropzone"] [data-testid="stFileDropzoneInstructions"] {{
            display: none !important; /* Cache l'icône cloud-upload + tout bloc d'instructions natif */
        }}
        [data-testid="stFileUploadDropzone"] section > div {{
            display: flex !important;
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 6px !important;
            position: relative !important;
        }}
        /* Rétablir UNIQUEMENT le bouton et le small (taille + couleur lisibles) */
        [data-testid="stFileUploadDropzone"] button {{
            font-size: 0.86rem !important;
            color: #E4E7EC !important;
        }}
        [data-testid="stFileUploadDropzone"] small {{
            font-size: 0.72rem !important;
            color: #848E9C !important;
        }}
        [data-testid="stFileUploadDropzone"] button {{
            color: #E4E7EC !important;
            position: relative !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.35rem !important;
            width: auto !important;
            max-width: 100% !important;
            flex: 0 0 auto !important;
            box-sizing: border-box !important;
            min-height: 2.25rem !important;
            padding: 0.4rem 0.85rem !important;
            overflow: visible !important;
        }}
        [data-testid="stFileUploadDropzone"] section > div > small {{
            display: block !important;
            position: relative !important;
            max-width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            font-size: 0.72rem !important;
            line-height: 1.4 !important;
            color: #848E9C !important;
            white-space: normal !important;
            word-break: break-word !important;
            visibility: visible !important;
        }}

        /* App — fond type TradeVizion */
        .stApp {{ background: #0b0e14 !important; color: #ECEEF4; font-family: Inter, Roboto, "Segoe UI", sans-serif !important; }}
        .main .block-container,
        section.main > div.block-container,
        [data-testid="stAppViewContainer"] .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }}
        [data-testid="stHorizontalBlock"] {{
            gap: 0.5rem !important;
        }}
        .tv-card {{
            background: #161a25;
            border: 1px solid #2a2e39;
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 10px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.28);
        }}
        /* Finance cards — ligne Dashboard (TradeVizion) */
        .tv-card.tv-card-dash-finance {{
            background: #161A25 !important;
            border: 1px solid #2A2E39 !important;
            border-radius: 10px !important;
            padding: 20px !important;
            text-align: center !important;
            margin-bottom: 10px !important;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35) !important;
        }}
        .tv-card-dash-finance__label {{
            color: #848E9C !important;
            font-size: 0.65rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            margin-bottom: 12px !important;
            line-height: 1.35 !important;
        }}
        .tv-card-dash-finance__value {{
            font-size: 2rem !important;
            font-weight: 800 !important;
            font-family: ui-monospace, "Courier New", Consolas, monospace !important;
            color: #FFFFFF !important;
            line-height: 1.15 !important;
            font-variant-numeric: tabular-nums !important;
        }}
        .tv-card-dash-finance__value--profit {{
            color: #00FFA3 !important;
            text-shadow:
                0 0 12px rgba(0, 255, 163, 0.55),
                0 0 28px rgba(0, 255, 163, 0.28) !important;
        }}
        .tv-card-dash-finance__value--loss {{
            color: #FF4B4B !important;
            text-shadow:
                0 0 10px rgba(255, 75, 75, 0.45),
                0 0 22px rgba(255, 75, 75, 0.22) !important;
        }}
        .tv-card-dash-finance__value--dd-ok {{
            color: #FFFFFF !important;
        }}
        .tv-card-dash-finance__value--dd-warn {{
            color: #FCA5A5 !important;
        }}
        .tv-card-dash-finance__suffix {{
            font-size: 0.55em !important;
            font-weight: 600 !important;
            color: #848E9C !important;
            margin-left: 0.12em !important;
        }}
        /* Glassmorphism — cartes Dashboard (TradeVizion / SaaS) */
        .matsa-dash-card {{
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            background: #161a25 !important;
            border: 1px solid #2a2e39 !important;
            border-radius: 8px;
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }}
        .matsa-dash-card--pad {{ padding: 12px 14px; }}
        .matsa-dash-card--badges {{ padding: 10px 12px; }}
        .matsa-dash-card--ring {{ padding: 10px 8px 12px; }}
        .matsa-dash-glory {{
            position: relative;
            padding: 11px 18px 15px;
            margin-bottom: 16px;
            background: rgba(6, 8, 12, 0.88);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            font-size: 0.8rem;
            color: #8B92A8;
            letter-spacing: 0.02em;
            overflow: hidden;
        }}
        .matsa-dash-glory__body {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
            gap: 6px 14px;
        }}
        .matsa-dash-glory__flow {{
            position: absolute;
            left: 10px;
            right: 10px;
            bottom: 4px;
            height: 2px;
            border-radius: 3px;
            background: linear-gradient(
                90deg,
                transparent 0%,
                rgba(0, 255, 163, 0.12) 18%,
                rgba(0, 255, 163, 0.95) 42%,
                rgba(0, 200, 255, 0.85) 58%,
                rgba(0, 255, 163, 0.25) 82%,
                transparent 100%
            );
            background-size: 220% 100%;
            animation: matsaGloryFlow 2.6s linear infinite;
            box-shadow: 0 0 12px rgba(0, 255, 163, 0.35);
        }}
        @keyframes matsaGloryFlow {{
            0% {{ background-position: 0% 50%; }}
            100% {{ background-position: -220% 50%; }}
        }}
        .matsa-dash-glory strong {{ color: #ECEEF4; font-weight: 700; }}
        .matsa-dash-glory__sep {{ color: #3D4456; padding: 0 4px; user-select: none; }}
        .matsa-dash-glory__profit {{ font-variant-numeric: tabular-nums; }}
        .matsa-dash-glory__profit--up {{ color: #00FFA3 !important; }}
        .matsa-dash-glory__profit--down {{ color: #FF6B6B !important; }}

        /* —— TradeVizion clone (Dashboard) —— */
        [data-testid="stVerticalBlock"] > div {{
            overflow: visible !important;
        }}
        .tvz-dist-head {{
            color: #6B7280;
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            text-align: left;
            margin-bottom: 10px;
        }}
        .tvz-hdr-card {{
            background: #161a25;
            border: 1px solid #2a2e39;
            border-radius: 8px;
            padding: 20px 0.8rem 1.35rem !important;
            text-align: center;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }}
        .tvz-hdr-card__label,
        .tvz-hdr-card__value {{
            display: block !important;
            line-height: 1.4 !important;
            padding: 2px 0 !important;
            overflow: visible !important;
        }}
        .tvz-hdr-card__label {{
            color: #FFFFFF;
            font-size: 0.58rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }}
        .tvz-hdr-card__value {{
            font-size: 1.6rem !important;
            font-weight: 800;
            font-variant-numeric: tabular-nums;
            font-family: ui-monospace, Consolas, monospace;
        }}
        .tvz-hdr-card__sub {{
            margin-top: auto;
            padding-top: 5px;
            font-size: 0.62rem;
            color: #6B7280;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
            opacity: 0.5;
            line-height: 1.2 !important;
            overflow: visible !important;
        }}
        .tvz-header-zone {{
            margin-top: 50px !important;
            margin-bottom: 20px !important;
            overflow: visible !important;
        }}
        .tvz-dashboard-title {{
            font-size: 2.5rem !important;
            font-weight: 800 !important;
            color: white !important;
            line-height: 1.2 !important;
            margin-bottom: 0.25rem;
        }}
        .tvz-dashboard-subtitle {{
            color: #6B7280;
            font-family: Inter, sans-serif;
            font-size: 0.9rem;
            font-weight: 500;
            margin: 0 0 1rem 0;
        }}
        .tvz-val--profit {{ color: #00FFA3 !important; }}
        .tvz-val--loss {{ color: #FF6B6B !important; }}
        .tvz-val--muted {{ color: #ECEEF4 !important; }}
        .tvz-hdr-card--green {{ border-top: 3px solid #00FFA3; }}
        .tvz-hdr-card--purple {{ border-top: 3px solid #A78BFA; }}
        .tvz-hdr-card--cyan {{ border-top: 3px solid #22D3EE; }}
        .tvz-hdr-card--blue {{ border-top: 3px solid #3B82F6; }}
        .tvz-hdr-card--yellow {{ border-top: 3px solid #EAB308; }}
        .tvz-hdr-card--orange {{ border-top: 3px solid #F97316; }}
        .tvz-score-card {{ padding: 14px 12px 16px !important; }}
        .tvz-score-mid {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 14px;
        }}
        .tvz-score-side {{
            flex: 0 0 58px;
            text-align: center;
            padding: 8px 5px 9px;
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.28);
        }}
        .tvz-score-side--perf {{
            border: 1px solid #2a2e39;
            border-left: 4px solid #00FFA3;
            box-shadow: 0 0 16px rgba(0, 255, 163, 0.18);
        }}
        .tvz-score-side--psycho {{
            border: 1px solid #2a2e39;
            border-right: 4px solid #FF2E63;
            box-shadow: 0 0 16px rgba(255, 46, 99, 0.16);
        }}
        .tvz-score-side__lbl {{
            font-size: 0.5rem;
            font-weight: 700;
            color: #6B7280;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        .tvz-score-side__score {{ line-height: 1.05; }}
        .tvz-score-side__nv {{
            font-size: 1.05rem;
            font-weight: 800;
            font-variant-numeric: tabular-nums;
        }}
        .tvz-score-side--perf .tvz-score-side__nv {{ color: #00FFA3; }}
        .tvz-score-side--psycho .tvz-score-side__nv {{ color: #FF2E63; }}
        .tvz-score-side__sf {{
            font-size: 0.58rem;
            font-weight: 600;
            color: #6B7280;
            margin-left: 1px;
        }}
        .tvz-score-center {{ flex: 1 1 auto; text-align: center; min-width: 0; }}
        .tvz-score-big {{
            font-size: 3.5rem !important;
            font-weight: 800;
            line-height: 1;
            color: #00FFA3;
            letter-spacing: -0.02em;
        }}
        .tvz-score-suf {{
            font-size: 1.05rem;
            font-weight: 700;
            color: #848E9C;
            margin-left: 2px;
        }}
        .tvz-score-lbl {{
            margin-top: 5px;
            font-size: 0.68rem;
            font-weight: 700;
            color: #EAB308;
        }}
        .tvz-score-hint {{
            margin-top: 3px;
            font-size: 0.58rem;
            font-weight: 500;
            color: #6B7280;
            line-height: 1.25;
        }}
        .tvz-score-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
            margin-top: 2px;
        }}
        .tvz-score-cell {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 7px 9px;
            font-size: 0.68rem;
            border-right: 1px solid #2a2e39;
            border-bottom: 1px solid #2a2e39;
            background: transparent;
        }}
        .tvz-score-cell:nth-child(2n) {{ border-right: none; }}
        .tvz-score-cell:nth-child(n+5) {{ border-bottom: none; }}
        .tvz-score-cell__k {{ color: #9CA3AF; font-weight: 600; }}
        .tvz-score-cell__v {{
            font-weight: 800;
            font-variant-numeric: tabular-nums;
        }}
        .tvz-frac--good {{ color: #00FFA3 !important; }}
        .tvz-frac--mid {{ color: #F59E0B !important; }}
        .tvz-frac--bad {{ color: #EF4444 !important; }}
        .tvz-section-title {{
            color: #6B7280;
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            text-align: left;
            margin: 12px 0 8px 0;
        }}
        .mastery-ring-thin-wrap.tvz-gauge-compact.matsa-dash-card {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
        }}
        .tvz-gauge-compact {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            padding: 8px 6px 10px !important;
            max-width: 120px;
            margin-left: auto;
            margin-right: auto;
        }}
        .tvz-gauge-svg-wrap {{
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            min-height: 52px;
        }}
        .tvz-gauge-svg-wrap--dash {{
            min-height: 48px;
            align-items: center;
        }}
        .tvz-gauge-val {{
            font-size: 0.95rem;
            font-weight: 800;
            font-variant-numeric: tabular-nums;
            color: #ECEEF4;
            text-align: center;
            line-height: 1.15;
            margin-top: 2px;
        }}
        .tvz-gauge-lbl,
        .tvz-gauge-compact .mastery-ring-thin-label {{
            margin-top: 4px !important;
            font-size: 0.55rem !important;
            color: #6B7280 !important;
            letter-spacing: 1px !important;
            font-weight: 600;
            text-transform: uppercase;
            text-align: center;
            line-height: 1.25;
        }}
        .tvz-ring-dash .tvz-ring-dash__circle {{
            font-size: 1.35rem;
            font-weight: 800;
            color: #6B7280;
            text-align: center;
            line-height: 1;
        }}
        .st-key-matsa_dash_objectifs {{
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            background: #161a25 !important;
            border: 1px solid #2a2e39 !important;
            border-radius: 12px;
            padding: 12px 14px 14px;
            margin: 10px 0 12px 0;
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }}
        .matsa-dash-goal-badge {{
            display: inline-block;
            margin-top: 10px;
            padding: 5px 12px;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #061015;
            background: linear-gradient(90deg, #00FFA3, #2EE6A8, #00C2FF);
            box-shadow: 0 0 16px rgba(0, 255, 163, 0.45);
        }}
        .matsa-dash-insight {{
            display: flex;
            gap: 10px;
            align-items: flex-start;
            padding: 12px 14px;
            margin-top: 10px;
        }}
        .matsa-dash-insight__icon {{
            flex: 0 0 auto;
            font-size: 1.05rem;
            line-height: 1.2;
            opacity: 0.95;
        }}
        .matsa-dash-insight__txt {{
            flex: 1 1 auto;
            font-size: 0.78rem;
            line-height: 1.48;
            color: #D4D8E3;
        }}
        .st-key-matsa_spark_exec,
        .st-key-matsa_spark_psycho {{
            margin-top: 2px !important;
        }}
        .st-key-matsa_spark_exec [data-testid="stVegaLiteChart"] .vega-embed .role-axis,
        .st-key-matsa_spark_exec [data-testid="stArrowVegaLiteChart"] .vega-embed .role-axis,
        .st-key-matsa_spark_psycho [data-testid="stVegaLiteChart"] .vega-embed .role-axis,
        .st-key-matsa_spark_psycho [data-testid="stArrowVegaLiteChart"] .vega-embed .role-axis {{
            display: none !important;
        }}
        .st-key-matsa_spark_exec [data-testid="stVegaLiteChart"] .vega-embed .role-title,
        .st-key-matsa_spark_exec [data-testid="stArrowVegaLiteChart"] .vega-embed .role-title,
        .st-key-matsa_spark_psycho [data-testid="stVegaLiteChart"] .vega-embed .role-title,
        .st-key-matsa_spark_psycho [data-testid="stArrowVegaLiteChart"] .vega-embed .role-title {{
            display: none !important;
        }}
        .st-key-matsa_spark_exec [data-testid="stVegaLiteChart"],
        .st-key-matsa_spark_exec [data-testid="stArrowVegaLiteChart"],
        .st-key-matsa_spark_psycho [data-testid="stVegaLiteChart"],
        .st-key-matsa_spark_psycho [data-testid="stArrowVegaLiteChart"] {{
            margin-top: -6px !important;
        }}
        .mastery-ring-thin-wrap {{ width: 100%; }}
        .mastery-ring-thin-label {{
            color: #8B92A8;
            font-size: 0.56rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 5px;
            text-align: center;
            line-height: 1.25;
        }}
        .matsa-trade-card-list {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 10px;
        }}
        .matsa-trade-card {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            padding: 10px 14px;
            border-radius: 11px;
            min-height: 52px;
        }}
        .matsa-trade-card__left {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex: 0 1 auto;
            min-width: 0;
        }}
        .matsa-trade-card__dir {{
            font-size: 0.78rem;
            font-weight: 900;
            line-height: 1;
        }}
        .matsa-trade-card__dir--buy {{
            color: #00FFA3;
            text-shadow: 0 0 10px rgba(0, 255, 163, 0.35);
        }}
        .matsa-trade-card__dir--sell {{
            color: #FF5A6D;
            text-shadow: 0 0 10px rgba(255, 90, 109, 0.3);
        }}
        .matsa-trade-card__sym {{
            font-weight: 700;
            font-size: 0.92rem;
            color: #F3F4F6;
            letter-spacing: 0.02em;
        }}
        .matsa-trade-card__center {{
            flex: 1 1 auto;
            text-align: center;
            font-size: 0.72rem;
            color: #9CA3AF;
        }}
        .matsa-trade-card__right {{ flex: 0 0 auto; }}
        .matsa-trade-card__badge {{
            display: inline-block;
            padding: 6px 11px;
            border-radius: 10px;
            font-weight: 800;
            font-size: 0.95rem;
            font-variant-numeric: tabular-nums;
            font-family: ui-monospace, "Courier New", Consolas, monospace;
        }}
        .matsa-trade-card__badge--win {{
            background: rgba(0, 255, 163, 0.12);
            color: #00FFA3;
            border: 1px solid rgba(0, 255, 163, 0.28);
        }}
        .matsa-trade-card__badge--loss {{
            background: rgba(255, 75, 90, 0.12);
            color: #FF6B6B;
            border: 1px solid rgba(255, 75, 90, 0.28);
        }}
        /* KPI mini — ligne 4 colonnes Dashboard */
        .tv-dash-kpi-mini {{
            background: transparent !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 14px 12px !important;
            margin-bottom: 8px !important;
            box-shadow: none !important;
            text-align: center !important;
        }}
        .tv-dash-kpi-mini__label {{
            color: #848E9C !important;
            font-size: 0.62rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
            margin-bottom: 8px !important;
            line-height: 1.3 !important;
        }}
        .tv-dash-kpi-mini .tv-card-dash-finance__value {{
            font-size: 1.42rem !important;
        }}
        .tv-dash-kpi-mini__delta {{
            color: #6B7280 !important;
            font-size: 0.66rem !important;
            font-weight: 600 !important;
            margin-top: 6px !important;
            letter-spacing: 0.02em !important;
            font-variant-numeric: tabular-nums !important;
        }}
        .tv-dash-badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 0;
            justify-content: center;
        }}
        .tv-dash-badges--empty {{
            justify-content: flex-start;
        }}
        .tv-dash-badges__muted {{
            color: #5C6370;
            font-size: 0.65rem;
        }}
        .tv-dash-badge {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 5px 10px;
            border-radius: 999px;
            border: 1px solid #2A2E39;
            background: rgba(22, 26, 37, 0.85);
            font-size: 0.72rem;
            color: #E5E7EB;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
        }}
        .tv-dash-badge__txt {{
            font-weight: 600;
            letter-spacing: 0.02em;
        }}
        /* Distribution sorties — barres horizontales */
        .tv-dash-exit-wrap {{
            margin: 6px 0 14px 0;
        }}
        .tv-dash-exit-head {{
            color: #A1A1AA;
            font-size: 0.65rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 10px;
        }}
        .tv-dash-exit-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            font-size: 0.72rem;
        }}
        .tv-dash-exit-row__name {{
            flex: 0 0 78px;
            color: #9CA3AF;
            font-weight: 600;
        }}
        .tv-dash-exit-row__track {{
            flex: 1 1 auto;
            height: 8px;
            background: #1E222D;
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid #2A2E39;
        }}
        .tv-dash-exit-row__fill {{
            height: 100%;
            border-radius: 4px;
            min-width: 0;
            transition: width 0.35s ease;
        }}
        .tv-dash-exit-row__n {{
            flex: 0 0 28px;
            text-align: right;
            color: #E5E7EB;
            font-variant-numeric: tabular-nums;
            font-weight: 700;
        }}
        /* Highlights bas de page Dashboard */
        .tv-dash-highlight {{
            background: transparent !important;
            border: none !important;
            border-radius: 10px;
            padding: 12px 14px;
            box-shadow: none !important;
            min-height: 72px;
        }}
        .tv-dash-highlight__emoji {{
            font-size: 0.84rem;
            line-height: 1;
            margin-bottom: 4px;
        }}
        .tv-dash-highlight__lbl {{
            color: #848E9C;
            font-size: 0.6rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}
        .tv-dash-highlight__val {{
            font-size: 1.05rem;
            font-weight: 800;
            font-family: ui-monospace, "Courier New", Consolas, monospace;
            color: #FFFFFF;
            font-variant-numeric: tabular-nums;
        }}
        .tv-dash-highlight__val--up {{ color: #00FFA3 !important; }}
        .tv-dash-highlight__val--down {{ color: #FF4B4B !important; }}
        .tv-title {{ color: #A1A1AA; font-size: 0.69rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.55px; }}
        .tv-value {{ color: #FFFFFF; font-size: 1.62rem; font-weight: 800; }}
        .setup-recap-table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; margin-top: 6px; }}
        .setup-recap-table th {{
            text-align: left; color: #A1A1AA; font-weight: 600; font-size: 0.65rem;
            text-transform: uppercase; letter-spacing: 0.5px; padding: 0.55rem 0.45rem 0.45rem 0;
            border-bottom: 1px solid #1F1F24;
        }}
        .setup-recap-table td {{ color: #E5E7EB; padding: 0.55rem 0.45rem; border-bottom: 1px solid #18181f; }}
        .setup-recap-table tr:last-child td {{ border-bottom: none; }}
        .sizing-assistant-card .tv-title {{ margin-bottom: 8px; }}
        .sizing-line-wrap {{
            font-size: 1rem; color: #D1D5DB; margin: 14px 0 6px 0; font-weight: 600;
        }}
        .sizing-risk-caption {{ font-size: 0.82rem; color: #9CA3AF; margin: 0 0 2px 0; line-height: 1.45; }}
        @keyframes matsaSizingPulseGlow {{
            0%, 100% {{
                filter: drop-shadow(0 0 6px rgba(var(--matsa-accent-rgb), 0.5)) drop-shadow(0 0 16px rgba(var(--matsa-accent-rgb), 0.28));
                transform: scale(1) translateZ(0);
            }}
            50% {{
                filter: drop-shadow(0 0 14px rgba(var(--matsa-accent-rgb), 0.95)) drop-shadow(0 0 32px rgba(var(--matsa-accent-rgb), 0.42));
                transform: scale(1.03) translateZ(0);
            }}
        }}
        .sizing-pulse-value {{
            display: inline-block;
            font-size: 1.95rem;
            font-weight: 800;
            color: var(--matsa-accent);
            font-variant-numeric: tabular-nums;
            letter-spacing: 0.02em;
            text-shadow:
                0 0 12px rgba(var(--matsa-accent-rgb), 0.85),
                0 0 28px rgba(var(--matsa-accent-rgb), 0.55),
                0 0 52px rgba(var(--matsa-accent-rgb), 0.28);
            animation: matsaSizingPulseGlow 2.1s ease-in-out infinite;
            will-change: filter, transform;
        }}
        .sizing-assistant-foot {{
            font-size: 0.72rem;
            color: #6B7280;
            margin: 10px 0 0 0;
            line-height: 1.35;
        }}
        .nt-trade-sizing-hero {{
            text-align: center !important;
            font-size: 1.75rem !important;
            font-weight: 900 !important;
            color: #E5E7EB !important;
            margin: 16px 0 6px 0 !important;
            line-height: 1.2 !important;
        }}
        .nt-trade-sizing-hero span.nt-sz-val {{
            font-family: ui-monospace, "Courier New", Consolas, monospace !important;
            font-size: 1.08em !important;
        }}
        .nt-trade-sizing-hero span.nt-sz-val--na {{
            color: #848E9C !important;
            text-shadow: none !important;
        }}
        .nt-trade-sizing-hero span.nt-sz-val--num {{
            color: #00FFA3 !important;
            text-shadow: 0 0 14px rgba(0, 255, 163, 0.45) !important;
        }}
        .trade-form-card.tv-card {{
            background: #161A25 !important;
            border: 1px solid #2A2E39 !important;
            border-radius: 10px !important;
            padding: 16px 18px !important;
            margin-bottom: 12px !important;
        }}
        /* Slider émotion — thumb & plage active vert néon #00FFA3 */
        div.st-key-nt_emotion_slider [data-baseweb="thumb"],
        div.st-key-nt_emotion_slider div[role="slider"] {{
            background-color: #00FFA3 !important;
            border: 2px solid #0a0a0a !important;
            box-shadow: 0 0 12px rgba(0, 255, 163, 0.55) !important;
        }}
        div.st-key-nt_emotion_slider [data-testid="stSlider"] [data-baseweb="thumb"],
        div.st-key-nt_emotion_slider [data-testid="stSlider"] div[role="slider"] {{
            background-color: #00FFA3 !important;
            border: 2px solid #0a0a0a !important;
            box-shadow: 0 0 12px rgba(0, 255, 163, 0.55) !important;
        }}
        div.st-key-nt_emotion_slider [data-baseweb="slider"] [data-baseweb="track"] > div:last-child,
        div.st-key-nt_emotion_slider [data-testid="stSlider"] [data-baseweb="slider"] [data-baseweb="track"] > div:last-child {{
            background: #00FFA3 !important;
            height: 4px !important;
            border-radius: 999px !important;
        }}
        .pnl-glow {{ color: var(--matsa-accent); text-shadow: 0 0 14px rgba(var(--matsa-accent-rgb), 0.35); }}
        .tvs-badge {{
            background: radial-gradient(circle at 30% 30%, rgba(99,102,241,0.46), rgba(99,102,241,0.16));
            border: 1px solid #6366F1;
            border-radius: 999px;
            width: 80px;
            height: 80px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto;
            color: #FFFFFF;
            font-size: 1.45rem;
            font-weight: 800;
        }}

        [data-testid="stExpander"] {{
            border: 1px solid #1F2937 !important;
            border-radius: 8px !important;
            background: #111827 !important;
        }}
        [data-testid="stExpander"] summary {{ color: #FFFFFF !important; font-size: 14px !important; }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: linear-gradient(180deg, #121218 0%, #101015 100%) !important;
            border: 1px solid #1F1F24 !important;
            border-radius: 12px !important;
            padding: 14px 16px 18px 16px !important;
            margin-bottom: 14px !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.28) !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSlider"] [data-baseweb="track"],
        div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSlider"] [data-testid="stSliderTrack"] {{
            height: 4px !important;
            border-radius: 999px !important;
            background: #1c1c22 !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSlider"] [data-baseweb="thumb"],
        div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSlider"] div[role="slider"] {{
            height: 14px !important;
            width: 14px !important;
            background-color: var(--matsa-accent) !important;
            border: 2px solid #0a0a0a !important;
            box-shadow: 0 0 10px rgba(var(--matsa-accent-rgb), 0.45) !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSlider"] [data-baseweb="slider"] [data-baseweb="track"] > div:last-child {{
            background: var(--matsa-accent) !important;
            height: 4px !important;
            border-radius: 999px !important;
        }}

        form[data-testid="stForm"] .stFormSubmitButton > button {{
            background-color: #000000 !important;
            color: #FFFFFF !important;
            border: 1px solid #3F3F46 !important;
            font-weight: 600 !important;
        }}
        form[data-testid="stForm"] .stFormSubmitButton > button:hover {{
            border-color: var(--matsa-accent) !important;
            box-shadow: 0 0 14px rgba(var(--matsa-accent-rgb), 0.45) !important;
            color: #FFFFFF !important;
        }}

        /* Paramètres — bouton Sauvegarder (rouge) */
        div.st-key-settings_save_btn button {{
            width: 100% !important;
            min-height: 48px !important;
            background: linear-gradient(180deg, #7f1d1d 0%, #450a0a 100%) !important;
            color: #ffffff !important;
            border: 1px solid #991b1b !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
        }}
        div.st-key-settings_save_btn button:hover {{
            border-color: #ef4444 !important;
            box-shadow: 0 0 16px rgba(239, 68, 68, 0.45) !important;
            color: #ffffff !important;
        }}

        /* Chrome Streamlit — pas d'icônes SVG décoratives dans la barre supérieure / marges */
        [data-testid="stHeader"] svg,
        [data-testid="stToolbar"] svg,
        [data-testid="stDecoration"] svg,
        [data-testid="stDeployButton"] svg,
        [data-testid="stMainMenu"] svg {{
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
        }}

        /* Modal graphique trade — bordure néon + lueur (overlay isolé : logo / horloges inchangés) */
        [data-testid="stDialog"] [data-testid="stImage"] {{
            border: 1px solid var(--matsa-accent) !important;
            border-radius: 6px !important;
            padding: 4px !important;
            background: #050505 !important;
            box-shadow:
                0 0 28px rgba(var(--matsa-accent-rgb), 0.42),
                0 0 72px rgba(var(--matsa-accent-rgb), 0.14) !important;
        }}
        [data-testid="stDialog"] [data-testid="stImage"] img,
        [data-testid="stDialog"] [data-testid="stImage"] picture img {{
            border-radius: 4px !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

all_trades = st.session_state.df.copy()
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

# Navigation : libellés courts + migration depuis anciennes valeurs / page Paramètres supprimée
_NAV_PAGES: tuple[str, ...] = (
    "Dashboard",
    "Suivi Mensuel",
    "News",
    "Stats",
    "Analyses",
    "Mon Trading",
    "Mon Compte",
    "Nouveau Trade",
)
_LEGACY_MAIN_NAV = {
    "Calendrier": "Suivi Mensuel",
    "News Economiques": "News",
    "Mes Stats": "Stats",
    "Analyses Avancées": "Analyses",
    "Mon Compte/Finance": "Mon Compte",
    "Parametres": "Mon Compte",
}
_TVZ_NAV_ROWS: tuple[tuple[str, str], ...] = (
    ("Dashboard", "dash"),
    ("Suivi Mensuel", "suivi"),
    ("News", "news"),
    ("Stats", "stats"),
    ("Analyses", "analyses"),
    ("Mon Trading", "trading"),
    ("Mon Compte", "compte"),
    ("Nouveau Trade", "nouveau"),
)
if "page" not in st.session_state:
    _leg = st.session_state.get("main_nav", "Dashboard")
    if _leg in _LEGACY_MAIN_NAV:
        _leg = _LEGACY_MAIN_NAV[_leg]
    st.session_state.page = _leg if _leg in _NAV_PAGES else "Dashboard"
if st.session_state.get("page") not in _NAV_PAGES:
    st.session_state.page = "Dashboard"

with st.sidebar:
    st.sidebar.markdown('<div class="matsa-logo">Mat\'Sa</div>', unsafe_allow_html=True)

    compte_options = ["Tous les comptes"]
    if account_names:
        compte_options += account_names
    _current_compte = str(st.session_state.get("global_compte_filter", "Tous les comptes"))
    if _current_compte not in compte_options:
        if st.session_state.get("pending_trade_compte") in compte_options:
            st.session_state["global_compte_filter"] = st.session_state.get("pending_trade_compte")
        elif account_names:
            st.session_state["global_compte_filter"] = account_names[0]
        else:
            st.session_state["global_compte_filter"] = "Tous les comptes"
    selected_compte = st.selectbox("Sélectionner un Compte", compte_options, key="global_compte_filter")

    with st.expander("➕ Ajouter un compte", expanded=False):
        sidebar_new_acc_name = st.text_input("Nom du nouveau compte", key="sidebar_new_acc_name")
        sidebar_new_acc_profit = st.number_input(
            "% de Profit Objectif",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
            key="sidebar_new_acc_profit",
        )
        sidebar_new_acc_max_loss = st.number_input(
            "Max Daily Loss (en $)",
            min_value=1.0,
            value=500.0,
            step=25.0,
            key="sidebar_new_acc_max_loss",
        )
        if st.button("Créer ce compte", key="sidebar_new_acc_btn", type="secondary"):
            clean_acc = str(sidebar_new_acc_name).strip()
            if not clean_acc:
                st.warning("Le nom du compte est obligatoire.")
            else:
                upsert_account(clean_acc, float(sidebar_new_acc_profit), float(sidebar_new_acc_max_loss), 10000.0)
                load_accounts_from_sheet.clear()
                st.session_state.pending_trade_compte = clean_acc
                st.session_state["global_compte_filter"] = clean_acc
                st.success(f"Compte « {clean_acc} » créé.")
                st.rerun()

    matsa_sidebar_clocks()
    st.markdown(
        '<div class="matsa-sidebar-clocks-spacer" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="tvz-nav-intro">Navigation</p>', unsafe_allow_html=True)
    for _lab, _slug in _TVZ_NAV_ROWS:
        _is_act = str(st.session_state.get("page", "")) == _lab
        if st.button(
            _lab,
            key=f"tvznav_{_slug}",
            use_container_width=True,
            type="primary" if _is_act else "secondary",
        ):
            st.session_state.page = _lab
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<p class="matsa-sidebar-import-head">Importer un trade (CSV, PDF, Image)</p>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Importer depuis TradingView (CSV, PDF, JPG, PNG)",
        type=["csv", "pdf", "jpg", "png"],
        key="tv_import",
        label_visibility="collapsed",
        help="1 Go maximum par fichier. CSV pour l'import automatique des trades.",
    )
    if uploaded_file is not None:
        fname = (uploaded_file.name or "").lower()
        ext = fname.rsplit(".", 1)[-1] if "." in fname else ""
        if ext != "csv":
            st.warning(
                "L'import automatique des trades nécessite un export CSV TradingView. "
                "Les fichiers PDF et images ne sont pas convertis pour le moment."
            )
        else:
            try:
                raw_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            except Exception:
                try:
                    uploaded_file.seek(0)
                    raw_df = pd.read_csv(
                        uploaded_file,
                        sep=None,
                        engine="python",
                        encoding="utf-8-sig",
                        on_bad_lines="skip",
                    )
                except Exception:
                    uploaded_file.seek(0)
                    raw_df = pd.read_csv(uploaded_file, sep=";", encoding="utf-8-sig", on_bad_lines="skip")
            try:
                converted = convert_tradingview_to_mvizion(raw_df)
                if converted.empty:
                    st.error("Aucun trade valide trouve dans le fichier.")
                else:
                    st.success(f"Importation de {append_trades(converted)} trades reussie.")
                    st.session_state.df = load_trades(time.time())
                    st.cache_data.clear()
                    st.rerun()
            except Exception:
                st.error("Echec importation TradingView. Verifier le CSV.")

    # Injection JS uploader : uniquement ici (sidebar), jamais dans un formulaire.
    matsa_sidebar_upload_translate_inject()

page = str(st.session_state.get("page", "Dashboard"))

# Filtre global : toutes les vues analytiques utilisent ``trades`` (sauf besoins explicites du contraire).
trades = filter_trades_par_compte(all_trades, selected_compte)

if page == "Nouveau Trade":
    st.subheader("Nouveau Trade")
    st.session_state["_sizing_capital_cached"] = _sizing_capital_for_ui(
        selected_compte, account_settings, account_names
    )
    if "sizing_frag_entry" not in st.session_state:
        st.session_state.sizing_frag_entry = 0.0
    if "sizing_frag_sl" not in st.session_state:
        st.session_state.sizing_frag_sl = 0.0
    if "nt_trade_risk_pct" not in st.session_state:
        st.session_state.nt_trade_risk_pct = float(st.session_state.get("risk_per_trade_pct", 1.0))
    _f_to_nt = (
        ("f_symbol", "nt_symbol"),
        ("f_quantite", "nt_quantite"),
        ("f_strategie", "nt_strategie"),
        ("f_emotion_slider", "nt_emotion_slider"),
        ("f_trade_date", "nt_trade_date"),
        ("f_trade_type", "nt_trade_type"),
        ("f_compte", "nt_compte"),
        ("f_compte_type", "nt_compte_type"),
        ("f_session", "nt_session"),
        ("f_profit_objectif_pct", "nt_profit_objectif_pct"),
        ("f_max_daily_loss_usd", "nt_max_daily_loss_usd"),
        ("f_prix_tp", "nt_prix_tp"),
        ("f_frais", "nt_frais"),
        ("f_sortie", "nt_sortie"),
        ("f_screenshot", "nt_screenshot"),
    )
    for _fk, _nk in _f_to_nt:
        if _nk not in st.session_state and _fk in st.session_state:
            st.session_state[_nk] = st.session_state[_fk]
    if "nt_quantite" not in st.session_state and "trade_quantite" in st.session_state:
        st.session_state.nt_quantite = float(st.session_state.get("trade_quantite") or 0.0)
    if "nt_quantite" not in st.session_state:
        st.session_state.nt_quantite = 0.0
    if "nt_symbol" not in st.session_state and "trade_symbole_journal" in st.session_state:
        st.session_state.nt_symbol = str(st.session_state.get("trade_symbole_journal") or "NAS100")
    if "nt_symbol" not in st.session_state:
        st.session_state.nt_symbol = str(st.session_state.get("trade_actif", "NAS100"))
    strat_opts = load_trade_strategy_options()
    _cur_st = st.session_state.get("nt_strategie")
    if _cur_st is not None and _cur_st not in strat_opts:
        st.session_state.nt_strategie = strat_opts[0]

    with st.container(border=True):
        st.markdown("### Score d'exécution (0-20)")
        ex1, ex2 = st.columns(2)
        with ex1:
            v_sizing = st.slider("Sizing", 0, 20, 10, key="v8_slider_sizing")
            v_overtrading = st.slider("Over-trading", 0, 20, 0, key="v8_slider_overtrading")
            v_sl = st.slider("Gestion SL", 0, 20, 10, key="v8_slider_sl")
        with ex2:
            v_bias = st.slider("Coherence Biais", 0, 20, 10, key="v8_slider_bias")
            v_revenge = st.slider("Controle Revenge", 0, 20, 0, key="v8_slider_revenge")

        exec_global = (
            float(v_sizing)
            + float(v_overtrading)
            + float(v_sl)
            + float(v_bias)
            + float(v_revenge)
        ) / 5.0
        if exec_global > 15:
            exec_color = _accent
        elif exec_global >= 10:
            exec_color = "#FFA500"
        else:
            exec_color = "#FF4B4B"
        st.markdown(
            f'<p style="margin:10px 0 14px 0;font-size:1.05rem;font-weight:600;color:{exec_color};">'
            f"Score Global : {exec_global:.1f}/20</p>",
            unsafe_allow_html=True,
        )
        st.number_input(
            "High_Water_Mark",
            min_value=0.0,
            value=0.0,
            step=10.0,
            key="v8_high_water_mark",
        )
        st.file_uploader(
            "Capture d'écran du graphique",
            type=["png", "jpg", "jpeg"],
            key="nt_screenshot",
        )

    _trade_sizing_assistant_ui()
    _cap_sync = float(st.session_state.get("_sizing_capital_cached", 10000.0))
    _pe_sync = float(st.session_state.get("sizing_frag_entry", 0.0))
    _ps_sync = float(st.session_state.get("sizing_frag_sl", 0.0))
    _rk_sync = float(st.session_state.get("nt_trade_risk_pct", 1.0))
    _inst_sync = str(st.session_state.get("trade_actif", "NAS100")).strip() or "NAS100"
    _lot_sync, _ = calculer_lots(_cap_sync, _rk_sync, _pe_sync, _ps_sync, _inst_sync)
    _sig_sync = (round(_pe_sync, 6), round(_ps_sync, 6), round(_rk_sync, 4), _inst_sync.upper(), round(_cap_sync, 2))
    if _sig_sync != st.session_state.get("_nt_sizing_sync_sig"):
        st.session_state.nt_symbol = _inst_sync
        if abs(_pe_sync - _ps_sync) < 1e-12:
            st.session_state.nt_quantite = 0.0
        elif _lot_sync > 0:
            st.session_state.nt_quantite = round(float(_lot_sync), 2)
        st.session_state["_nt_sizing_sync_sig"] = _sig_sync

    st.markdown('<div class="tv-card trade-form-card">', unsafe_allow_html=True)
    st.markdown("#### Saisie du trade", unsafe_allow_html=True)
    _popover = getattr(st, "popover", None)
    if callable(_popover):
        with _popover("➕ Nouvelle Stratégie"):
            st.caption("Ajout persistant (fichier à côté des réglages UI).")
            _nw = st.text_input("Nom de la stratégie", key="nt_strategy_new_input", placeholder="Ex. FVG, Opening range…")
            if st.button("Ajouter à la liste", key="nt_strategy_add_btn"):
                _ok, _msg, _name_canon = append_trade_strategy(_nw)
                if _ok:
                    st.session_state.nt_strategie = _name_canon
                    st.success(_msg)
                    st.rerun()
                else:
                    st.warning(_msg)
    else:
        with st.expander("➕ Nouvelle Stratégie", expanded=False):
            _nw2 = st.text_input("Nouvelle stratégie", key="nt_strategy_new_input_fallback")
            if st.button("Ajouter", key="nt_strategy_add_btn_fallback"):
                _ok2, _msg2, _name2 = append_trade_strategy(_nw2)
                if _ok2:
                    st.session_state.nt_strategie = _name2
                    st.success(_msg2)
                    st.rerun()
                else:
                    st.warning(_msg2)

    with st.form("trade_form", clear_on_submit=False):
        st.select_slider(
            "Note psychologique (émotion)",
            options=list(NT_EMOTION_SLIDER_OPTIONS),
            value=NT_EMOTION_SLIDER_OPTIONS[0],
            key="nt_emotion_slider",
        )
        rd1, rd2 = st.columns([1.15, 1.0])
        with rd1:
            trade_date = st.date_input("Date", value=date.today(), key="nt_trade_date")
        with rd2:
            trade_type = st.radio("Direction", ["BUY", "SELL"], horizontal=True, key="nt_trade_type")
        top_left, top_right = st.columns([1.3, 1.1])
        with top_left:
            strategie_utilisee = st.selectbox(
                "Stratégie utilisée",
                strat_opts,
                key="nt_strategie",
            )
            trade_account_options = account_names.copy() if account_names else ["Compte 1"]
            pending = str(st.session_state.get("pending_trade_compte", "")).strip()
            _p_can = _compte_canonical_from_list(trade_account_options, pending) if pending else None
            selected_index = trade_account_options.index(_p_can) if _p_can is not None else 0
            compte = st.selectbox("Compte", trade_account_options, index=selected_index, key="nt_compte")
            if pending:
                st.session_state.pending_trade_compte = ""
            compte_type = st.selectbox("Type de Compte", ["Evaluation", "Funded", "Live"], key="nt_compte_type")
            session = st.selectbox("Session", ["AUTO", "ASIA", "LONDON", "NEW YORK", "OUT"], index=0, key="nt_session")
            _acc_k = _compte_settings_key(account_settings, compte) or compte
            default_acc = account_settings.get(_acc_k, {"profit_pct": 10.0, "max_daily_loss_usd": 500.0})
            profit_objectif_pct = st.number_input(
                "% de Profit Objectif (compte)",
                min_value=1.0,
                max_value=100.0,
                value=float(default_acc["profit_pct"]),
                step=0.5,
                key="nt_profit_objectif_pct",
            )
            max_daily_loss_usd_trade = st.number_input(
                "Max Daily Loss (en $)",
                min_value=1.0,
                value=float(default_acc["max_daily_loss_usd"]),
                step=25.0,
                key="nt_max_daily_loss_usd",
            )
        with top_right:
            st.text_input(
                "Symbole (journal)",
                key="nt_symbol",
                placeholder="ex: NAS100, US30...",
                help="Pré-rempli depuis le calculateur ; modifiable avant enregistrement.",
            )
            prix_tp = st.number_input("TP", min_value=0.0, value=0.0, step=0.01, key="nt_prix_tp")
            quantite = st.number_input(
                "Quantite (lots)",
                min_value=0.0,
                step=0.01,
                key="nt_quantite",
                help="Pré-rempli depuis le calculateur de sizing.",
            )
            frais = st.number_input("Frais", min_value=0.0, value=0.0, step=0.01, key="nt_frais")
            sortie = st.selectbox("Sortie", ["SL", "TP", "BE", "TP Partiel"], key="nt_sortie")
        _sym_ready = str(st.session_state.get("nt_symbol", "")).strip()
        try:
            _qty_ready = float(st.session_state.get("nt_quantite", 0.0) or 0.0)
        except (TypeError, ValueError):
            _qty_ready = 0.0
        _form_trade_ready = bool(_sym_ready) and _qty_ready > 1e-12
        if not _form_trade_ready:
            st.caption(
                "Indique un **symbole** (journal) et une **quantité** strictement supérieure à 0 "
                "pour activer le bouton « Ajouter Trade »."
            )
        try:
            submit = st.form_submit_button(
                "Ajouter Trade", use_container_width=True, disabled=not _form_trade_ready
            )
        except TypeError:
            try:
                submit = st.form_submit_button(
                    "Ajouter Trade", width="stretch", disabled=not _form_trade_ready
                )
            except TypeError:
                submit = st.form_submit_button("Ajouter Trade", disabled=not _form_trade_ready)
    st.markdown("</div>", unsafe_allow_html=True)
    if submit:
        actif = (
            str(st.session_state.get("nt_symbol", st.session_state.get("trade_actif", "NAS100")))
            .strip()
            .upper()
            or "NAS100"
        )
        prix_entree = float(st.session_state.get("sizing_frag_entry", 0.0))
        prix_sl = float(st.session_state.get("sizing_frag_sl", 0.0))
        sizing_score = float(st.session_state.get("v8_slider_sizing", 10))
        overtrading_score = float(st.session_state.get("v8_slider_overtrading", 0))
        sl_score = float(st.session_state.get("v8_slider_sl", 10))
        bias_score = float(st.session_state.get("v8_slider_bias", 10))
        revenge_score = float(st.session_state.get("v8_slider_revenge", 0))
        exec_global_save = (sizing_score + overtrading_score + sl_score + bias_score + revenge_score) / 5.0
        high_water_mark_saisie = float(st.session_state.get("v8_high_water_mark", 0.0))
        shot = st.session_state.get("nt_screenshot")
        emotion_save = _trade_emotion_canonical_from_session()

        if not str(st.session_state.get("nt_symbol", "")).strip():
            st.error("Le symbole (journal) est obligatoire.")
        elif float(quantite) <= 0:
            st.error("La quantité doit être strictement supérieure à zéro.")
        elif prix_entree <= 0.0:
            st.error("Indique un prix d'entrée valide dans le bloc « Pré-saisie — sizing temps réel ».")
        else:
            setup_to_save = str(strategie_utilisee).strip()
            profit = (prix_tp - prix_entree) * quantite - frais
            etat_mental = infer_mental_state(sizing_score, sl_score, revenge_score, overtrading_score, bias_score)
            trade_id = f"{trade_date.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}"
            image_rel = save_screenshot(shot, trade_id) if shot else ""
            _ck_save = _compte_settings_key(account_settings, compte)
            existing_cap = float(
                account_settings.get(_ck_save, {}).get("initial_capital", 10000.0) if _ck_save else 10000.0
            )
            upsert_account(compte, float(profit_objectif_pct), float(max_daily_loss_usd_trade), existing_cap)
            save_trade(
                {
                    "Date": pd.to_datetime(trade_date).strftime("%Y-%m-%d"),
                    "Actif": str(actif).strip().upper(),
                    "Type": trade_type,
                    "Setup": setup_to_save,
                    "Prix Entree": float(prix_entree),
                    "Prix Sortie": float(prix_tp),
                    "Quantite": float(quantite),
                    "Frais": float(frais),
                    "Profit": float(profit),
                    "Sortie": sortie,
                    "Session": get_trading_session() if session == "AUTO" else session,
                    "Etat Mental": etat_mental,
                    "Emotion": emotion_save,
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
                    "Execution_Score_Global": float(exec_global_save),
                    "High_Water_Mark_Saisie": float(high_water_mark_saisie),
                    "Image": image_rel,
                }
            )
            st.session_state.df = load_trades()
            st.rerun()

elif page == "Dashboard":
    st.markdown('<div class="tvz-header-zone"><h1 class="tvz-dashboard-title">Dashboard</h1></div>', unsafe_allow_html=True)
    st.markdown('<p class="tvz-dashboard-subtitle">Vue d\'ensemble de vos performances</p>', unsafe_allow_html=True)
    if all_trades.empty and trades_papa_csv_is_effectively_empty():
        st.error(
            "Le fichier trades_papa.csv est vide ou illisible : aucun trade à afficher. "
            "Vérifie l'export ou la synchronisation Google Sheet, puis rafraîchis la page."
        )
    dm = dash_metriques_anneaux(trades)
    eq = equite_series_dashboard(trades)
    pnl_per = dash_pnl_par_periode(trades)
    qd = dash_quarter_pnls(trades)
    pt = float(eq["profit_total"])
    peak_last = float(eq["peak"][-1]) if eq.get("peak") else pt
    eq_last = float(eq["equity"][-1]) if eq.get("equity") else pt
    dh = eq_last - peak_last

    h1, h2, h3, h4, h5, h6 = st.columns(6)
    with h1:
        st.markdown(
            _tvz_header_card_html(
                "BALANCE",
                html.escape(f"${pt:,.2f}"),
                "tvz-val--profit" if pt >= 0 else "tvz-val--loss",
                html.escape(_tvz_money_signed(float(pnl_per["delta_profit_total"]))),
                "green",
            ),
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            _tvz_header_card_html(
                "PEAK",
                html.escape(f"${peak_last:,.2f}"),
                "tvz-val--muted",
                html.escape(f"DH: ${dh:,.2f}"),
                "purple",
            ),
            unsafe_allow_html=True,
        )
    _td = float(pnl_per["today"])
    _tcls_td = "tvz-val--profit" if _td >= 0 else "tvz-val--loss"
    with h3:
        st.markdown(
            _tvz_header_card_html(
                "AUJOURD'HUI",
                html.escape(_tvz_money_signed(_td)),
                _tcls_td,
                html.escape(f"Préc: {_tvz_money_signed(float(pnl_per['yesterday']))}"),
                "cyan",
            ),
            unsafe_allow_html=True,
        )
    _wk = float(pnl_per["week"])
    with h4:
        st.markdown(
            _tvz_header_card_html(
                "SEMAINE",
                html.escape(_tvz_money_signed(_wk)),
                "tvz-val--profit" if _wk >= 0 else "tvz-val--loss",
                html.escape(f"Préc: {_tvz_money_signed(float(pnl_per['week_prev']))}"),
                "blue",
            ),
            unsafe_allow_html=True,
        )
    _mo = float(pnl_per["month"])
    with h5:
        st.markdown(
            _tvz_header_card_html(
                "MOIS",
                html.escape(_tvz_money_signed(_mo)),
                "tvz-val--profit" if _mo >= 0 else "tvz-val--loss",
                html.escape(f"Préc: {_tvz_money_signed(float(pnl_per['month_prev']))}"),
                "yellow",
            ),
            unsafe_allow_html=True,
        )
    _qt = float(qd["quarter"])
    with h6:
        st.markdown(
            _tvz_header_card_html(
                "TRIMESTRE",
                html.escape(_tvz_money_signed(_qt)),
                "tvz-val--profit" if _qt >= 0 else "tvz-val--loss",
                html.escape(f"Préc: {_tvz_money_signed(float(qd['quarter_prev']))}"),
                "orange",
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<div class="matsa-dash-rings-gap" aria-hidden="true"></div>', unsafe_allow_html=True)

    left, mid, right = st.columns([1, 2.5, 1.5])
    with left:
        if trades.empty:
            st.caption("Aucun trade pour cette sélection.")
        else:
            st.markdown(
                _dashboard_exit_distribution_html(dash_sorties_distribution_counts(trades)),
                unsafe_allow_html=True,
            )
    with mid:
        st.markdown(
            _tvz_score_center_html(dash_tradevizion_score_widget(trades)),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(_tvz_section_title("badges"), unsafe_allow_html=True)
        st.markdown(_dashboard_badges_html(matsa_badges_logic(trades)), unsafe_allow_html=True)
        st.markdown(_tvz_section_title("objectifs"), unsafe_allow_html=True)
        with st.container(key="matsa_dash_objectifs"):
            _goal_key = f"matsa_dash_goal_text_{date.today().isoformat()}"
            st.text_input("Objectif du jour", placeholder="Fixer un objectif…", key=_goal_key)
            _mdl_thr = _dashboard_max_daily_loss_usd(selected_compte, account_settings, account_names)
            _today_p = float(pnl_per["today"])
            if _mdl_thr > 0 and _today_p >= _mdl_thr:
                st.markdown(
                    '<div class="matsa-dash-goal-badge">Objectif atteint !</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="matsa-dash-rings-gap" aria-hidden="true"></div>', unsafe_allow_html=True)

    gr1, gr2, gr3, gr4 = st.columns(4)
    _wr = float(dm["win_rate"])
    _pf = float(dm["profit_factor"])
    if not math.isfinite(_pf):
        _pf = 0.0
    _n_l = int(dm["losses"])
    _n_w = int(dm["wins"])
    with gr1:
        st.markdown(
            generate_metric_ring_thin("WIN RATE", _wr, 60.0, "%"),
            unsafe_allow_html=True,
        )
    with gr2:
        if _n_l == 0 and not trades.empty and _n_w > 0:
            st.markdown(_tvz_ring_dash_placeholder("PROFIT FACTOR"), unsafe_allow_html=True)
        else:
            st.markdown(
                generate_metric_ring_thin("PROFIT FACTOR", _pf, 3.0, ""),
                unsafe_allow_html=True,
            )
    with gr3:
        _arr = float(dm["avg_rr"])
        if _n_l == 0 and not trades.empty and _n_w > 0:
            st.markdown(_tvz_ring_dash_placeholder("AVG R:R"), unsafe_allow_html=True)
        else:
            st.markdown(
                generate_metric_ring_thin("AVG R:R", min(_arr, 10.0), 4.0, "R"),
                unsafe_allow_html=True,
            )
    with gr4:
        st.markdown(
            generate_metric_ring_thin("JOURNÉES GAGNANTES", float(dm["win_day_pct"]), 75.0, "%"),
            unsafe_allow_html=True,
        )

    st.markdown("### Highlights")
    if trades.empty:
        best_v = 0.0
        worst_v = 0.0
    else:
        _prof = pd.to_numeric(trades["Profit"], errors="coerce").fillna(0.0)
        best_v = float(_prof.max())
        worst_v = float(_prof.min())

    hi1, hi2, hi3, hi4 = st.columns(4)
    _b_cls = (
        "tv-dash-highlight__val--up"
        if best_v > 0
        else ("tv-dash-highlight__val--down" if best_v < 0 else "")
    )
    _w_cls = (
        "tv-dash-highlight__val--up"
        if worst_v > 0
        else ("tv-dash-highlight__val--down" if worst_v < 0 else "")
    )
    with hi1:
        st.markdown(
            _dashboard_highlight_card("🏆", "Meilleur trade ($)", f"${best_v:,.2f}", _b_cls),
            unsafe_allow_html=True,
        )
    with hi2:
        st.markdown(
            _dashboard_highlight_card("💀", "Pire trade ($)", f"${worst_v:,.2f}", _w_cls),
            unsafe_allow_html=True,
        )
    with hi3:
        st.markdown(
            _dashboard_highlight_card("⏱️", "Plus long trade", "—"),
            unsafe_allow_html=True,
        )
    with hi4:
        st.markdown(
            _dashboard_highlight_card("📊", "Durée moyenne", "—"),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    if trades.empty:
        st.info("Aucun trade pour cette sélection.")
    else:
        b_eq, b_act = st.columns([2, 1])
        with b_eq:
            st.caption("Performance cumulée")
            fig_eq = equity_curve_figure(trades, _accent, height=280, paper_bg="#0b0e14")
            st.plotly_chart(fig_eq, theme=None, config=dict(displayModeBar=False), width="stretch")
        with b_act:
            st.caption("Trades récents")
            _rows_act = dash_derniers_trades_rows(trades, 5)
            if not _rows_act:
                st.caption("Aucune activité récente.")
            else:
                st.markdown(
                    '<div class="matsa-trade-card-list">'
                    + "".join(matsa_trade_card(r) for r in _rows_act)
                    + "</div>",
                    unsafe_allow_html=True,
                )

elif page == "Suivi Mensuel":
    st.subheader("Suivi Mensuel")
    st.caption("Filtre actif : compte choisi dans la sidebar (ou tous les comptes).")
    st.markdown("### Heatmap de performance (profit mensuel)")
    fig_mois = performance_monthly_heatmap_figure(trades)
    st.plotly_chart(fig_mois, theme=None, width="stretch")

elif page == "News":
    st.subheader("News économiques")
    st.markdown("### News économiques - Investing.com (FR)")
    st.caption("Filtre US à fort impact.")
    st.iframe(
        "https://fr.investing.com/economic-calendar/",
        height=520,
        width="stretch",
    )
    st.markdown("### News économiques - Forex Factory")
    st.caption("Source complémentaire (US impact fort).")
    st.iframe(
        "https://www.dailyfx.com/economic-calendar",
        height=520,
        width="stretch",
    )

elif page == "Stats":
    st.subheader("Stats — Statistiques détaillées")
    if trades.empty:
        st.info("Aucune statistique disponible.")
    else:
        st.markdown("### Discipline Score")
        _ds_pct = discipline_score_sur_100(trades)
        st.markdown(
            discipline_progress_ring_html(_ds_pct, _accent, _accent_rgb),
            unsafe_allow_html=True,
        )
        mental_stats = trades.groupby("Etat Mental", as_index=False)["Profit"].mean().sort_values("Profit", ascending=False)
        mental_stats["Couleur"] = mental_stats["Etat Mental"].map(ETAT_MENTAL_COLORS).fillna("#D1D5DB")
        fig_mental = go.Figure(data=[go.Bar(x=mental_stats["Etat Mental"], y=mental_stats["Profit"], marker_color=mental_stats["Couleur"])])
        fig_mental.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font={"color": "#FFFFFF"}, height=350)
        st.plotly_chart(fig_mental, theme=None, width="stretch")
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
        st.dataframe(table_df, width="stretch", hide_index=True)

elif page == "Analyses":
    st.subheader("Analyses avancées")
    if trades.empty:
        st.info("Aucun trade à analyser pour le compte sélectionné.")
    else:
        st.markdown("### Profil psychologique du trader")
        st.caption(
            "Moyenne des cinq scores d'exécution (0–20) sur les trades affichés : "
            "Sizing, Over-trading, gestion SL, cohérence de biais et contrôle Revenge. "
            "Source : mêmes données que le journal (feuille ou fichier local)."
        )
        fig_brain = trader_brain_radar_figure(trades, _accent)
        st.plotly_chart(fig_brain, theme=None, width="stretch")

        st.markdown("### Performance par stratégie")
        st.caption(
            "PnL cumulé = somme des profits par setup. La stratégie est stockée dans la colonne « Setup » "
            "(la direction du trade reste « Type » : BUY / SELL)."
        )
        stats_setup = performance_par_setup_agregat(trades)
        fig_setup = setup_pnl_bar_figure(stats_setup, _accent)
        st.plotly_chart(fig_setup, theme=None, config=dict(displayModeBar=False), width="stretch")
        rows_html = "".join(
            "<tr>"
            f"<td>{html.escape(str(r['Setup']))}</td>"
            f"<td>{float(r['Win_Rate_Pct']):.1f} %</td>"
            f"<td>${float(r['Profit_Net']):,.2f}</td>"
            f"<td>${float(r['Esperance']):,.2f}</td>"
            "</tr>"
            for _, r in stats_setup.iterrows()
        )
        st.markdown(
            '<div class="tv-card"><div class="tv-title">Synthèse par setup</div>'
            '<table class="setup-recap-table">'
            "<thead><tr>"
            "<th>Setup</th><th>Win rate</th><th>Profit net</th><th>Espérance ($ / trade)</th>"
            "</tr></thead><tbody>"
            f"{rows_html}"
            "</tbody></table></div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Analyse du drawdown (underwater)")
        st.caption(
            "Équité = PnL cumulé des trades affichés. "
            "Drawdown = (équité actuelle − sommet historique de l'équité) ÷ sommet historique. "
            "Zone rouge : distance au sommet (jusqu'à 0 % sur l'axe)."
        )
        fig_uw = underwater_drawdown_figure(trades)
        st.plotly_chart(fig_uw, theme=None, config=dict(displayModeBar=False), width="stretch")

        best_sess, risk_day = session_insights_paris_heatmap(trades)
        conseil = strategie_conseiller_message(trades)
        bs = html.escape(str(best_sess))
        rd = html.escape(str(risk_day))
        cq = html.escape(conseil)
        st.markdown(
            f'<div class="tv-card">'
            f'<div class="tv-title">Conseiller stratégique</div>'
            f'<p style="color:#D1D5DB;margin:0.3rem 0 0.65rem 0;font-size:0.95rem;line-height:1.5;">{cq}</p>'
            f'<hr style="border:none;border-top:1px solid #1F2937;margin:0.65rem 0;">'
            f'<div class="tv-title">Insights de session</div>'
            f'<p style="color:#E5E7EB;margin:0.35rem 0 0.2rem 0;font-size:1rem;">'
            f"Session la plus rentable : <strong>{bs}</strong></p>"
            f'<p style="color:#E5E7EB;margin:0.2rem 0 0.45rem 0;font-size:1rem;">'
            f"Jour le plus risqué : <strong>{rd}</strong></p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Analyse de Session (Heatmap)")
        st.caption(
            "Fuseau horaire : Europe/Paris. Matrice lundi–vendredi × 0h–23h ; "
            "chaque case est la somme des profits des trades dans ce créneau."
        )
        fig_session_paris = session_heatmap_paris_figure(trades, _accent)
        st.plotly_chart(fig_session_paris, theme=None, width="stretch")

        st.markdown("### Lien entre discipline et résultat")
        st.caption(
            "Le score de discipline est la moyenne des cinq critères d’exécution (0–20). "
            "La ligne violette est une tendance statistique : elle aide à visualiser si les trades plus disciplinés coïncident avec de meilleurs profits."
        )
        fig_disc, corr = discipline_profit_correlation_figure(trades)
        st.plotly_chart(fig_disc, theme=None, width="stretch")
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
        row_sel, row_btn = st.columns([3.2, 1])
        with row_sel:
            selected = st.selectbox(
                "Sélectionner un trade (détail, graphique, suppression)",
                options=options,
                key="delete_trade_select",
            )
        sel_pos = options.index(selected)
        selected_row = sorted_df.iloc[sel_pos]
        selected_index = int(selected_row["index"])
        img_resolved = resolve_trade_screenshot_path(str(selected_row.get("Image", "") or ""))
        with row_btn:
            st.markdown("<div style='height:1.65rem;'></div>", unsafe_allow_html=True)
            if img_resolved:
                if st.button("Voir graphique", key="mon_trading_voir_graphique_selection", width="stretch"):
                    trade_chart_dialog(img_resolved)
            else:
                st.caption("Pas de visuel")
        if not img_resolved:
            st.markdown(
                '<p style="color:#9CA3AF;font-size:0.9rem;margin:0.25rem 0 0.75rem 0;">'
                "Aucun visuel disponible pour ce trade."
                "</p>",
                unsafe_allow_html=True,
            )
        if st.button("Supprimer ce trade"):
            delete_trade_by_position(selected_index)
            st.success("Trade supprimé.")
            st.rerun()
        display_df = trades.sort_values("Date", ascending=False).copy()
        display_df["Visuel"] = display_df["Image"].apply(
            lambda x: "Oui" if resolve_trade_screenshot_path(str(x or "")) else "—"
        )
        display_df["Date"] = display_df["Date"].apply(format_date_fr)
        if "Emotion" not in display_df.columns:
            display_df["Emotion"] = "Calme"
        ordered_cols = [
            "Date",
            "Actif",
            "Session",
            "Type",
            "Setup",
            "Prix Entree",
            "Prix Sortie",
            "Quantite",
            "Frais",
            "Profit",
            "Sortie",
            "Etat Mental",
            "Emotion",
            "Compte",
            "Compte_Type",
            "Visuel",
        ]
        ordered_cols = [c for c in ordered_cols if c in display_df.columns]
        st.dataframe(display_df[ordered_cols], width="stretch", hide_index=True)

        viz_actions: list[tuple[str, str]] = []
        for pos, (_, r) in enumerate(sorted_df.iterrows()):
            rp = resolve_trade_screenshot_path(str(r.get("Image", "") or ""))
            if rp:
                dlab = format_date_fr(r["Date"])
                act = str(r.get("Actif", ""))[:18]
                viz_actions.append((f"{dlab} · {act}", rp))

        if viz_actions:
            st.markdown("##### Graphiques enregistrés")
            st.caption("Ouvre le graphique en plein écran (modal). La sidebar et le logo ne sont pas rechargés.")
            for batch in range(0, len(viz_actions), 4):
                chunk = viz_actions[batch : batch + 4]
                cols = st.columns(len(chunk))
                for i, (label, path) in enumerate(chunk):
                    with cols[i]:
                        if st.button(
                            "Voir graphique",
                            key=f"mon_trading_vizdlg_{batch}_{i}",
                            help=label,
                            width="stretch",
                        ):
                            trade_chart_dialog(path)

elif page == "Mon Compte":
    st.subheader("Mon Compte — Finance & réglages")
    with st.expander("Réglages Mat'Sa, feuille Accounts & comptes", expanded=False):
        _settings_panel_body()
    st.markdown("---")
    if not account_names:
        st.info("Aucun compte disponible. Crée d'abord un compte depuis la sidebar (➕ Ajouter un compte).")
    else:
        finance_compte = _compte_canonical_from_list(account_names, selected_compte) or account_names[0]
        if selected_compte == "Tous les comptes":
            st.caption("Sélectionne un compte précis dans la sidebar pour des chiffres totalement isolés par compte.")
        _fk = _compte_settings_key(account_settings, finance_compte) or finance_compte
        finance_settings = account_settings.get(
            _fk, {"profit_pct": 10.0, "max_daily_loss_usd": 500.0, "initial_capital": 10000.0}
        )
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

        if selected_compte != "Tous les comptes":
            trades_finance = trades.copy()
        else:
            trades_finance = filter_trades_par_compte(all_trades, finance_compte)
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
            width="stretch",
            hide_index=True,
        )
st.caption(f"Trades chargés : {len(st.session_state.df)}")
