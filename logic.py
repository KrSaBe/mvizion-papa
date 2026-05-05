import calendar
import html
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
import threading
import logging
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import gspread
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from gspread.exceptions import APIError, SpreadsheetNotFound


# --- Google Sheets (remplace l'ancien CSV) ---
# Configuration exclusivement via Streamlit secrets (.streamlit/secrets.toml) :
#   MVIZION_GOOGLE_SHEET_ID : ID du tableur (URL entre /d/ et /edit)
#   MVIZION_GOOGLE_WORKSHEET : (optionnel) nom de l'onglet, defaut "Trades"
#   [gcp_service_account] : champs du JSON compte de service Google (type, project_id, ...)
#
# Partage : le compte service (client_email) doit etre invite en "Editeur" sur le tableur.

COLUMNS = [
    "Date",
    "Actif",
    "Type",
    "Setup",
    "Prix Entree",
    "Prix Sortie",
    "Quantite",
    "Frais",
    "Profit",
    "Sortie",
    "Session",
    "Etat Mental",
    "Emotion",
    "Biais Jour",
    "Compte",
    "Compte_Type",
    "Profit_Objectif_Pct",
    "Max_Daily_Loss_USD",
    "Sizing_Score",
    "SL_Score",
    "Revenge_Score",
    "Overtrading_Score",
    "Bias_Score",
    "Execution_Score_Global",
    "High_Water_Mark_Saisie",
    "High_Water_Mark",
    "Image",
]
ACCOUNT_COLUMNS = ["Nom", "Objectif_Pct", "Max_Loss_USD", "Initial_Capital"]

# Fichier CSV local : chemin absolu basé sur le répertoire de travail (synchro avec Streamlit / Windows).
FILE_PATH = os.path.join(os.getcwd(), "trades_papa.csv")

# Préférences UI (police logo, couleur d'accent) — fichier JSON à la racine du projet
UI_SETTINGS_FILE = "matsa_ui_settings.json"
DEFAULT_UI_SETTINGS: dict[str, str] = {
    "primary_font": "Playfair Display",
    "accent_color": "#00FFA3",
    "risk_per_trade_pct": "1.0",
}


def _ui_settings_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), UI_SETTINGS_FILE)


def load_ui_settings() -> dict[str, str]:
    path = _ui_settings_path()
    if not os.path.isfile(path):
        return dict(DEFAULT_UI_SETTINGS)
    try:
        import json

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        out = dict(DEFAULT_UI_SETTINGS)
        if isinstance(data, dict):
            pf = str(data.get("primary_font", "")).strip()
            if pf:
                out["primary_font"] = pf
            ac = str(data.get("accent_color", "")).strip()
            if ac:
                out["accent_color"] = ac if ac.startswith("#") else f"#{ac}"
            rp = data.get("risk_per_trade_pct")
            if rp is not None and str(rp).strip() != "":
                try:
                    fv = float(str(rp).strip().replace(",", "."))
                    if fv > 0:
                        out["risk_per_trade_pct"] = str(max(0.1, min(5.0, fv)))
                except ValueError:
                    pass
        return out
    except Exception:
        logging.getLogger("mvizion.logic").exception("Lecture mat_sa_ui_settings impossible")
        return dict(DEFAULT_UI_SETTINGS)


def save_ui_settings(settings: dict[str, str]) -> None:
    import json

    path = _ui_settings_path()
    out = dict(DEFAULT_UI_SETTINGS)
    for k in DEFAULT_UI_SETTINGS:
        if k in settings and str(settings[k]).strip():
            out[k] = str(settings[k]).strip()
    if not out["accent_color"].startswith("#"):
        out["accent_color"] = "#" + out["accent_color"]
    try:
        rpv = float(str(out.get("risk_per_trade_pct", "1.0")).replace(",", "."))
        out["risk_per_trade_pct"] = str(max(0.1, min(5.0, rpv)))
    except ValueError:
        out["risk_per_trade_pct"] = DEFAULT_UI_SETTINGS["risk_per_trade_pct"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


# Stratégies personnalisées (journal) — persistance JSON à côté de ``matsa_ui_settings.json``
STRATEGIES_JSON_FILE = "matsa_custom_strategies.json"
DEFAULT_TRADE_STRATEGIES: tuple[str, ...] = ("Breakout", "Retest", "Reversal", "Scalp")


def _strategies_json_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), STRATEGIES_JSON_FILE)


def _read_strategy_extras_from_disk() -> list[str]:
    import json

    path = _strategies_json_path()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("strategies") if isinstance(data, dict) else data
        if not isinstance(raw, list):
            return []
        base_l = {x.lower() for x in DEFAULT_TRADE_STRATEGIES}
        out: list[str] = []
        seen: set[str] = set()
        for x in raw:
            s = str(x).strip()
            if not s or s.lower() in base_l or s.lower() == "autre":
                continue
            if s.lower() in seen:
                continue
            seen.add(s.lower())
            out.append(s)
        return out
    except Exception:
        return []


def load_trade_strategy_options() -> list[str]:
    """Liste des stratégies au sélecteur : base + entrées persistantes (sans « Autre »)."""
    out = list(DEFAULT_TRADE_STRATEGIES) + _read_strategy_extras_from_disk()
    return [x for x in out if str(x).strip().lower() != "autre"]


def append_trade_strategy(name: str) -> tuple[bool, str, str]:
    """Ajoute une stratégie hors liste de base ; persistance dans ``matsa_custom_strategies.json``.

    Retourne ``(succès, message, nom_canonique)`` ; ``nom_canonique`` est la chaîne enregistrée si succès, sinon ``""``.
    """
    import json

    clean = str(name or "").strip()
    if not clean:
        return False, "Nom vide.", ""
    if clean.lower() == "autre":
        return False, "Le libellé « Autre » n'est pas autorisé (stratégies via le popover).", ""
    base_l = {x.lower() for x in DEFAULT_TRADE_STRATEGIES}
    if clean.lower() in base_l:
        return False, "Identique à une stratégie de base.", ""
    opts = load_trade_strategy_options()
    if any(o.lower() == clean.lower() for o in opts):
        return False, "Déjà présente dans la liste.", ""
    extras = _read_strategy_extras_from_disk()
    extras.append(clean)
    path = _strategies_json_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"strategies": extras}, f, indent=2, ensure_ascii=False)
        return True, f"Stratégie « {clean} » enregistrée.", clean
    except OSError as exc:
        return False, f"Échec écriture fichier : {exc}", ""


def calculate_position_size(
    capital: float,
    risk_percent: float,
    entry: float,
    stop_loss: float,
    tick_value: float = 1.0,
) -> tuple[float, float]:
    """
    Taille de position (lots / contrats) pour risquer ``risk_percent`` % du capital.

    Retourne ``(lots, risque_monetaire_exact_usd)`` où le risque exact est ``lots * écart * tick_value``.
    Si le stop est égal au prix d'entrée (écart nul), retourne ``(0.0, 0.0)``.
    """
    if capital <= 0.0 or risk_percent <= 0.0 or tick_value <= 0.0:
        return 0.0, 0.0
    try:
        e = float(entry)
        s = float(stop_loss)
        tv = float(tick_value)
    except (TypeError, ValueError):
        return 0.0, 0.0
    distance = abs(e - s)
    if distance < 1e-12:
        return 0.0, 0.0
    target_risk = float(capital) * (float(risk_percent) / 100.0)
    dollar_per_lot = distance * tv
    if dollar_per_lot < 1e-12:
        return 0.0, 0.0
    lots = target_risk / dollar_per_lot
    actual_risk = lots * dollar_per_lot
    return float(lots), float(actual_risk)


def _instrument_usd_per_price_point(instrument: str) -> float:
    """
    Dollars risqués par **point de prix** et par **lot** (dénominateur de ``calculer_lots``).

    Formule générale : ``Lots = (Capital × (Risque % / 100)) / (|Entrée − SL| × USD/point)``.

    V10 : facteur **1** par défaut pour tous les symboles listés ; surcharge possible par indice
    (ex. NAS100, US30) lorsque tu fixeras un $/point réel.
    """
    s = str(instrument or "").strip().upper().replace(" ", "")
    if not s:
        return 1.0
    # Clés normalisées — toutes à 1.0 tant que les contrats $/point ne sont pas branchés sur données broker
    table: dict[str, float] = {
        "NAS100": 1.0,
        "MNQ1!": 1.0,
        "NQ": 1.0,
        "MES1!": 1.0,
        "MGC1!": 1.0,
        "DXY": 1.0,
        "US30": 1.0,
        "US500": 1.0,
        "GER40": 1.0,
    }
    for sym, usd in table.items():
        if s == sym or sym in s or s in sym:
            return float(usd)
    return 1.0


def calculer_lots(
    capital: float,
    risque_pct: float,
    prix_entree: float,
    prix_sl: float,
    instrument: str = "NAS100",
) -> tuple[float, float]:
    """
    Lots et risque monétaire ($) pour risquer ``risque_pct`` % du capital.

    ``Lots = (Capital × (Risque % / 100)) / (|prix_entree − prix_sl| × USD_par_point(instrument))``.

    Si entrée = stop (écart nul), retourne ``(0.0, 0.0)``.
    """
    if float(capital) <= 0.0 or float(risque_pct) <= 0.0:
        return 0.0, 0.0
    try:
        e = float(prix_entree)
        s = float(prix_sl)
    except (TypeError, ValueError):
        return 0.0, 0.0
    distance = abs(e - s)
    if distance < 1e-12:
        return 0.0, 0.0
    usd_pp = _instrument_usd_per_price_point(instrument)
    denom = distance * float(usd_pp)
    if denom < 1e-12:
        return 0.0, 0.0
    cible = float(capital) * (float(risque_pct) / 100.0)
    lots = cible / denom
    risque_usd = float(lots) * denom
    return float(lots), float(risque_usd)


def dash_metriques_anneaux(df: pd.DataFrame) -> dict[str, Any]:
    """
    Métriques pour les 4 anneaux du Dashboard : win rate, profit factor + libellé,
    R:R moyen (même logique que compute_metrics), % de journées gagnantes.
    """
    out: dict[str, Any] = {
        "win_rate": 0.0,
        "wins": 0,
        "losses": 0,
        "profit_factor": 0.0,
        "profit_factor_label": "—",
        "avg_rr": 0.0,
        "n_trades_rr": 0,
        "win_days": 0,
        "total_days": 0,
        "win_day_pct": 0.0,
    }
    if df.empty:
        return out
    profits = pd.to_numeric(df["Profit"], errors="coerce").fillna(0.0)
    n = int(len(profits))
    if n == 0:
        return out
    wins_mask = profits > 0
    losses_mask = profits < 0
    n_w = int(wins_mask.sum())
    n_l = int(losses_mask.sum())
    out["wins"] = n_w
    out["losses"] = n_l
    out["win_rate"] = float(wins_mask.mean() * 100.0)
    gross_profit = float(profits[wins_mask].sum()) if n_w else 0.0
    gross_loss = float(profits[losses_mask].sum()) if n_l else 0.0
    if gross_loss != 0.0:
        pf = gross_profit / abs(gross_loss)
    else:
        pf = gross_profit if gross_profit > 0 else 0.0
    out["profit_factor"] = float(pf)
    if pf >= 2.5:
        out["profit_factor_label"] = "Excellent"
    elif pf >= 1.75:
        out["profit_factor_label"] = "Très bon"
    elif pf >= 1.25:
        out["profit_factor_label"] = "Bon"
    elif pf >= 1.0:
        out["profit_factor_label"] = "Correct"
    elif pf > 0:
        out["profit_factor_label"] = "À améliorer"
    else:
        out["profit_factor_label"] = "—"

    avg_win = float(profits[wins_mask].mean()) if n_w else 0.0
    avg_loss_abs = float(profits[losses_mask].abs().mean()) if n_l else 0.0
    if avg_loss_abs > 0:
        avg_rr = avg_win / avg_loss_abs
    elif avg_win > 0:
        avg_rr = 99.99
    else:
        avg_rr = 0.0
    out["avg_rr"] = float(min(avg_rr, 99.99))
    out["n_trades_rr"] = n

    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work["Profit"] = pd.to_numeric(work["Profit"], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["Date"])
    if work.empty:
        return out
    daily = work.groupby(work["Date"].dt.normalize(), dropna=False)["Profit"].sum()
    total_days = int(len(daily))
    win_days = int((daily > 0).sum()) if total_days else 0
    out["total_days"] = total_days
    out["win_days"] = win_days
    out["win_day_pct"] = float(win_days / total_days * 100.0) if total_days else 0.0
    return out


def generate_metric_ring(label: str, value: float, target: float, unit: str = "") -> str:
    """
    Anneau SVG « Mastery » : remplissage ``min(value/target, 1.0)`` (borné à ≥ 0 si cible > 0).
    Trait **#FFD700** si ``value >= target``, sinon **#00FFA3**. Piste de fond **#1E222D**, ``stroke-width`` 8.
    """
    val_f = float(value)
    tgt_f = float(target)
    if tgt_f <= 0:
        fill_frac = 1.0 if val_f > 0 else 0.0
    else:
        fill_frac = max(0.0, min(val_f / tgt_f, 1.0))
    stroke = "#FFD700" if val_f >= tgt_f else "#00FFA3"
    cx, cy, r = 70.0, 70.0, 40.0
    sw = 8.0
    circ = 2.0 * math.pi * r
    offset = circ * (1.0 - fill_frac)

    if unit == "%":
        center_txt = f"{val_f:.1f}%"
    elif unit == "R":
        center_txt = f"{val_f:.2f} R"
    elif str(unit).strip():
        center_txt = f"{val_f:.2f} {str(unit).strip()}"
    else:
        center_txt = f"{val_f:.2f}"

    lbl_raw = str(label).strip()
    lbl_e = html.escape(lbl_raw)
    ctr_e = html.escape(center_txt)

    return (
        f'<div class="mastery-ring-wrap">'
        f'<svg class="mastery-ring-svg" width="100%" style="max-width:140px;height:auto;display:block;margin:0 auto;" '
        f'viewBox="0 0 140 140" xmlns="http://www.w3.org/2000/svg" role="img" shape-rendering="geometricPrecision" '
        f'aria-label="{lbl_e}">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#1E222D" stroke-width="{sw}"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
        f'stroke-linecap="round" stroke-dasharray="{circ:.4f}" stroke-dashoffset="{offset:.4f}" '
        f'transform="rotate(-90 {cx} {cy})"/>'
        f'<text x="70" y="78" text-anchor="middle" fill="#FFFFFF" font-size="17" font-weight="800" '
        f'font-family="ui-monospace,Courier New,Consolas,monospace">{ctr_e}</text>'
        f"</svg>"
        f'<div class="mastery-ring-label">{lbl_e}</div>'
        f"</div>"
    )


def generate_metric_ring_thin(label: str, value: float, target: float, unit: str = "") -> str:
    """Anneau SVG compact : valeur et libellé sous l'anneau (style TradeVizion)."""
    val_f = float(value)
    tgt_f = float(target)
    if tgt_f <= 0:
        fill_frac = 1.0 if val_f > 0 else 0.0
    else:
        fill_frac = max(0.0, min(val_f / tgt_f, 1.0))
    stroke = "#A78BFA" if val_f >= tgt_f else "#00FFA3"
    cx, cy, r = 44.0, 44.0, 30.0
    sw = 3.2
    circ = 2.0 * math.pi * r
    offset = circ * (1.0 - fill_frac)

    if unit == "%":
        center_txt = f"{val_f:.0f}%"
    elif unit == "R":
        center_txt = f"{val_f:.1f} R"
    elif str(unit).strip():
        center_txt = f"{val_f:.1f} {str(unit).strip()}"
    else:
        center_txt = f"{val_f:.0f}"

    lbl_raw = str(label).strip()
    lbl_e = html.escape(lbl_raw)
    ctr_e = html.escape(center_txt)

    return (
        f'<div class="mastery-ring-thin-wrap matsa-dash-card matsa-dash-card--ring tvz-gauge-compact">'
        f'<div class="tvz-gauge-svg-wrap">'
        f'<svg class="mastery-ring-thin-svg" width="100%" style="max-width:92px;height:auto;display:block;margin:0 auto;" '
        f'viewBox="0 0 88 88" xmlns="http://www.w3.org/2000/svg" role="img" shape-rendering="geometricPrecision" '
        f'aria-label="{lbl_e}">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#2a2e39" stroke-width="{sw}"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
        f'stroke-linecap="round" stroke-dasharray="{circ:.4f}" stroke-dashoffset="{offset:.4f}" '
        f'transform="rotate(-90 {cx} {cy})"/>'
        f"</svg></div>"
        f'<div class="tvz-gauge-val">{ctr_e}</div>'
        f'<div class="mastery-ring-thin-label tvz-gauge-lbl">{lbl_e}</div>'
        f"</div>"
    )


MONTHS_FR = {
    1: "Jan",
    2: "Fev",
    3: "Mar",
    4: "Avr",
    5: "Mai",
    6: "Jun",
    7: "Jul",
    8: "Aou",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

_sheet_lock = threading.RLock()
_gspread_client: gspread.Client | None = None
_gspread_spreadsheet: gspread.Spreadsheet | None = None
LOGGER = logging.getLogger("mvizion.logic")


def _show_google_error(message: str, technical: str = "") -> None:
    st.warning(message)
    if technical:
        st.caption(f"Détail technique: {technical}")


def _handle_google_exception(exc: Exception, context: str) -> None:
    LOGGER.exception("Google Sheets error during %s", context)
    technical = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, RuntimeError):
        _show_google_error(
            "Configuration Streamlit Secrets invalide ou incomplète. Vérifie `MVIZION_GOOGLE_SHEET_ID` et la section `[gcp_service_account]`.",
            technical,
        )
    if isinstance(exc, SpreadsheetNotFound):
        _show_google_error(
            "Google Sheet introuvable (404) ou inaccessible. Vérifie l'ID du sheet et le partage au compte de service.",
            technical,
        )
    if isinstance(exc, APIError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 429:
            _show_google_error(
                "Quota Google Sheets (429) dépassé. Réessaie plus tard ; les trades peuvent s'afficher depuis trades_papa.csv si le fichier est présent.",
                technical,
            )
            return
        if status == 403:
            _show_google_error(
                "Erreur de permissions Google (403 Forbidden). Partage le sheet avec le `client_email` du compte de service en Éditeur.",
                technical,
            )
        if status == 404:
            _show_google_error(
                "Google Sheet introuvable (404 Not Found). Vérifie `MVIZION_GOOGLE_SHEET_ID`.",
                technical,
            )
        _show_google_error(
            "Échec API Google Sheets. Vérifie les secrets Streamlit, l'ID du sheet et les autorisations.",
            technical,
        )
    _show_google_error(
        "Connexion Google Sheets impossible. Vérifie les secrets Streamlit et les permissions du compte de service.",
        technical,
    )


def _clear_cached_reads() -> None:
    try:
        st.cache_data.clear()
    except Exception:
        LOGGER.exception("Impossible de vider le cache Streamlit")


def _is_quota_error(exc: Exception) -> bool:
    if isinstance(exc, APIError):
        resp = getattr(exc, "response", None)
        code = getattr(resp, "status_code", None) if resp is not None else None
        if code == 429:
            return True
    low = str(exc).lower()
    return any(
        s in low
        for s in (
            "429",
            "quota",
            "resource_exhausted",
            "rate limit",
            "user_rate_limit",
        )
    )


def _trade_csv_primary_path() -> str:
    """Chemin absolu du CSV ``trades_papa.csv`` (``FILE_PATH``)."""
    return os.path.abspath(FILE_PATH)


def _trade_csv_paths() -> list[str]:
    """Ordre de recherche : dossier du module, puis répertoire courant."""
    return [_trade_csv_primary_path(), os.path.join(os.getcwd(), FILE_PATH)]


def _read_local_trades_dataframe() -> pd.DataFrame:
    for path in _trade_csv_paths():
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            try:
                df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", on_bad_lines="skip")
            except Exception:
                LOGGER.exception("Lecture locale impossible: %s", path)
                continue
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = _default_for_column(col)
        return df.reindex(columns=COLUMNS)
    return pd.DataFrame(columns=COLUMNS)


def sync_sheet_metadata() -> None:
    """À appeler au premier chargement ou après « Sauvegarder » : vérifie le Sheet puis vide le cache de lecture."""
    ensure_csv_exists()
    _clear_cached_reads()


def _service_account_dict_from_secrets() -> dict[str, Any]:
    try:
        if "gcp_service_account" not in st.secrets:
            raise RuntimeError(
                "Secrets Streamlit : ajouter la section [gcp_service_account] dans .streamlit/secrets.toml "
                "(champs du JSON compte de service)."
            )
        section = st.secrets["gcp_service_account"]
        payload = {str(k): section[k] for k in section}
        required = {"type", "project_id", "private_key", "client_email", "token_uri"}
        missing = [k for k in required if not str(payload.get(k, "")).strip()]
        if missing:
            raise RuntimeError(f"Champs secrets manquants: {', '.join(missing)}")
        return payload
    except Exception as exc:
        _handle_google_exception(exc, "load_service_account_secrets")
        raise


def _spreadsheet_id() -> str:
    try:
        sid = str(st.secrets.get("MVIZION_GOOGLE_SHEET_ID", "")).strip()
        if not sid or sid == "METS_TON_ID_ICI":
            raise RuntimeError(
                "Secrets Streamlit : definir MVIZION_GOOGLE_SHEET_ID dans .streamlit/secrets.toml "
                "(ID du tableur, segment entre /d/ et /edit dans l'URL)."
            )
        return sid
    except Exception as exc:
        _handle_google_exception(exc, "load_sheet_id")
        raise


def _worksheet_title() -> str:
    if "MVIZION_GOOGLE_WORKSHEET" in st.secrets:
        t = str(st.secrets["MVIZION_GOOGLE_WORKSHEET"] or "").strip()
        if t:
            return t
    return "Trades"


def _accounts_worksheet_title() -> str:
    return "Accounts"


def _get_client() -> gspread.Client:
    global _gspread_client
    try:
        if _gspread_client is None:
            info = _service_account_dict_from_secrets()
            _gspread_client = gspread.service_account_from_dict(info)
        return _gspread_client
    except Exception as exc:
        _handle_google_exception(exc, "create_gspread_client")
        raise


def _get_spreadsheet() -> gspread.Spreadsheet:
    global _gspread_spreadsheet
    try:
        if _gspread_spreadsheet is None:
            gc = _get_client()
            _gspread_spreadsheet = gc.open_by_key(_spreadsheet_id())
        return _gspread_spreadsheet
    except Exception as exc:
        _handle_google_exception(exc, "open_spreadsheet")
        raise


def _open_worksheet(title: str, default_rows: int = 2000) -> gspread.Worksheet:
    try:
        sh = _get_spreadsheet()
        try:
            return sh.worksheet(title)
        except gspread.WorksheetNotFound:
            return sh.add_worksheet(title=title, rows=default_rows, cols=max(len(COLUMNS), 26))
    except Exception as exc:
        _handle_google_exception(exc, f"open_worksheet:{title}")
        raise


def _open_trades_worksheet() -> gspread.Worksheet:
    return _open_worksheet(_worksheet_title(), default_rows=2000)


def _open_accounts_worksheet() -> gspread.Worksheet:
    return _open_worksheet(_accounts_worksheet_title(), default_rows=200)


def _ensure_accounts_sheet_exists() -> None:
    aws = _open_accounts_worksheet()
    a_raw = aws.get_all_values()
    if not a_raw:
        aws.update(
            [ACCOUNT_COLUMNS],
            range_name=f"A1:{_a1_column(len(ACCOUNT_COLUMNS))}1",
            value_input_option="USER_ENTERED",
            raw=False,
        )
        return
    a_header = [str(h).strip() for h in a_raw[0]]
    if a_header == ACCOUNT_COLUMNS:
        return
    rows = a_raw[1:] if len(a_raw) > 1 else []
    rebuilt = [ACCOUNT_COLUMNS]
    for r in rows:
        row = list(r) + [""] * (len(ACCOUNT_COLUMNS) - len(r))
        rebuilt.append(row[: len(ACCOUNT_COLUMNS)])
    aws.clear()
    aws.update(
        rebuilt,
        range_name=f"A1:{_a1_column(len(ACCOUNT_COLUMNS))}{max(1, len(rebuilt))}",
        value_input_option="USER_ENTERED",
        raw=False,
    )


def _values_to_dataframe(header: list[str], rows: list[list[Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    header = [str(h).strip() for h in header]
    width = max(len(header), max((len(r) for r in rows), default=0))
    header = (header + [""] * width)[:width]
    norm_rows: list[list[Any]] = []
    for r in rows:
        row = list(r) + [""] * (width - len(r))
        norm_rows.append(row[:width])
    df = pd.DataFrame(norm_rows, columns=header[:width])
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = _default_for_column(col)
    return df.reindex(columns=COLUMNS)


def _cell_value(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return v


def _dataframe_to_sheet_values(df: pd.DataFrame) -> list[list[Any]]:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    for c in COLUMNS:
        if c not in out.columns:
            out[c] = _default_for_column(c)
    out = out[COLUMNS]
    body: list[list[Any]] = []
    for _, row in out.iterrows():
        body.append([_cell_value(row[c]) for c in COLUMNS])
    return [COLUMNS] + body


def _read_sheet_dataframe() -> pd.DataFrame:
    ws = _open_trades_worksheet()
    raw = ws.get_all_values()
    if not raw:
        return pd.DataFrame(columns=COLUMNS)
    header, *data_rows = raw
    if not header or all(str(c).strip() == "" for c in header):
        return pd.DataFrame(columns=COLUMNS)
    df = _values_to_dataframe(header, data_rows)
    return df


def _mirror_trades_to_local_csv(df: pd.DataFrame) -> None:
    """Écrit une copie du journal dans ``FILE_PATH`` (UTF-8 BOM)."""
    path = _trade_csv_primary_path()
    try:
        out = df.copy()
        for col in COLUMNS:
            if col not in out.columns:
                out[col] = _default_for_column(col)
        out = out[COLUMNS]
        out.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception:
        LOGGER.exception("Écriture CSV locale impossible: %s", path)


def _write_sheet_dataframe(df: pd.DataFrame) -> None:
    ws = _open_trades_worksheet()
    values = _dataframe_to_sheet_values(df)
    ws.clear()
    last_col = _a1_column(len(COLUMNS))
    last_row = max(1, len(values))
    ws.update(values, range_name=f"A1:{last_col}{last_row}", value_input_option="USER_ENTERED", raw=False)
    _mirror_trades_to_local_csv(df)


def _a1_column(n: int) -> str:
    """1 -> A, 26 -> Z, 27 -> AA."""
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _default_for_column(col: str) -> Any:
    numeric_cols = {
        "Prix Entree",
        "Prix Sortie",
        "Quantite",
        "Frais",
        "Profit",
        "Sizing_Score",
        "SL_Score",
        "Revenge_Score",
        "Overtrading_Score",
        "Bias_Score",
        "Execution_Score_Global",
        "High_Water_Mark_Saisie",
        "High_Water_Mark",
        "Profit_Objectif_Pct",
        "Max_Daily_Loss_USD",
    }
    if col in numeric_cols:
        return 0.0
    if col == "Sortie":
        return "TP"
    if col == "Etat Mental":
        return "Neutre"
    if col == "Emotion":
        return "Calme"
    if col == "Biais Jour":
        return "Haussier"
    if col == "Compte":
        return "Compte 1"
    if col == "Compte_Type":
        return "Eval"
    if col == "Type":
        return "Buy"
    if col == "Setup":
        return ""
    if col == "Session":
        return "London"
    if col == "Image":
        return ""
    return ""


def ensure_csv_exists() -> None:
    """Initialise l'onglet Google Sheet (en-tetes COLUMNS) et migre les colonnes manquantes."""
    try:
        with _sheet_lock:
            _ensure_accounts_sheet_exists()
            ws = _open_trades_worksheet()
            raw = ws.get_all_values()
            if not raw:
                ws.update(
                    [COLUMNS],
                    range_name=f"A1:{_a1_column(len(COLUMNS))}1",
                    value_input_option="USER_ENTERED",
                    raw=False,
                )
                return
            header = [str(h).strip() for h in raw[0]]
            if header == COLUMNS:
                return
            data_rows = raw[1:] if len(raw) > 1 else []
            df = _values_to_dataframe(header, data_rows)
            for col in COLUMNS:
                if col not in df.columns:
                    df[col] = _default_for_column(col)
            df = df[COLUMNS]
            df = _apply_high_water_mark(df)
            _write_sheet_dataframe(df)
    except APIError as exc:
        if _is_quota_error(exc):
            LOGGER.warning("ensure_csv_exists: quota Google Sheets, synchronisation reportée.")
            return
        _handle_google_exception(exc, "ensure_csv_exists")
    except Exception as exc:
        _handle_google_exception(exc, "ensure_csv_exists")


@st.cache_data(ttl=300, show_spinner=False)
def load_accounts_from_sheet() -> pd.DataFrame:
    try:
        with _sheet_lock:
            ws = _open_accounts_worksheet()
            raw = ws.get_all_values()
    except Exception as exc:
        if _is_quota_error(exc):
            LOGGER.warning("load_accounts_from_sheet: quota / limite API, retour vide.")
        else:
            _handle_google_exception(exc, "load_accounts_from_sheet")
        return pd.DataFrame(columns=ACCOUNT_COLUMNS)
    if not raw:
        return pd.DataFrame(columns=ACCOUNT_COLUMNS)
    header, *rows = raw
    if not header:
        return pd.DataFrame(columns=ACCOUNT_COLUMNS)
    width = max(len(header), len(ACCOUNT_COLUMNS))
    header = (list(header) + [""] * width)[:width]
    normalized_rows: list[list[Any]] = []
    for r in rows:
        normalized_rows.append((list(r) + [""] * width)[:width])
    df = pd.DataFrame(normalized_rows, columns=header)
    for col in ACCOUNT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[ACCOUNT_COLUMNS]
    df["Nom"] = df["Nom"].astype(str).str.strip()
    df["Objectif_Pct"] = pd.to_numeric(df["Objectif_Pct"], errors="coerce").fillna(10.0)
    df["Max_Loss_USD"] = pd.to_numeric(df["Max_Loss_USD"], errors="coerce").fillna(500.0)
    df["Initial_Capital"] = pd.to_numeric(df["Initial_Capital"], errors="coerce").fillna(10000.0)
    df = df[df["Nom"] != ""].copy()
    df["_key"] = df["Nom"].str.lower()
    df = df.drop_duplicates("_key", keep="last").drop(columns=["_key"]).reset_index(drop=True)
    return df


def upsert_account(name: str, objectif_pct: float, max_loss_usd: float, initial_capital: float = 10000.0) -> None:
    clean = str(name).strip()
    if not clean:
        return
    with _sheet_lock:
        ws = _open_accounts_worksheet()
        raw = ws.get_all_values()
        if not raw:
            table = pd.DataFrame(columns=ACCOUNT_COLUMNS)
        else:
            header, *rows = raw
            width = max(len(header), len(ACCOUNT_COLUMNS))
            header = (list(header) + [""] * width)[:width]
            normalized_rows = [(list(r) + [""] * width)[:width] for r in rows]
            table = pd.DataFrame(normalized_rows, columns=header)
        for col in ACCOUNT_COLUMNS:
            if col not in table.columns:
                table[col] = ""
        table = table[ACCOUNT_COLUMNS].copy()
        table["Nom"] = table["Nom"].astype(str).str.strip()
        table["Objectif_Pct"] = pd.to_numeric(table["Objectif_Pct"], errors="coerce").fillna(10.0)
        table["Max_Loss_USD"] = pd.to_numeric(table["Max_Loss_USD"], errors="coerce").fillna(500.0)
        table["Initial_Capital"] = pd.to_numeric(table["Initial_Capital"], errors="coerce").fillna(10000.0)
        key = clean.lower()
        mask = table["Nom"].astype(str).str.lower() == key
        if mask.any():
            idx = table[mask].index[-1]
            table.at[idx, "Nom"] = clean
            table.at[idx, "Objectif_Pct"] = float(objectif_pct)
            table.at[idx, "Max_Loss_USD"] = float(max_loss_usd)
            table.at[idx, "Initial_Capital"] = float(initial_capital)
            # Supprime d'éventuels doublons résiduels
            table["_key"] = table["Nom"].astype(str).str.lower()
            table = table.drop_duplicates("_key", keep="last").drop(columns=["_key"])
        else:
            table = pd.concat(
                [
                    table,
                    pd.DataFrame(
                        [
                            {
                                "Nom": clean,
                                "Objectif_Pct": float(objectif_pct),
                                "Max_Loss_USD": float(max_loss_usd),
                                "Initial_Capital": float(initial_capital),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        values = [ACCOUNT_COLUMNS] + table[ACCOUNT_COLUMNS].values.tolist()
        ws.clear()
        ws.update(
            values,
            range_name=f"A1:{_a1_column(len(ACCOUNT_COLUMNS))}{max(1, len(values))}",
            value_input_option="USER_ENTERED",
            raw=False,
        )
    _clear_cached_reads()


def delete_account(name: str) -> bool:
    clean = str(name).strip()
    if not clean:
        return False
    with _sheet_lock:
        ws = _open_accounts_worksheet()
        raw = ws.get_all_values()
        if not raw:
            return False
        header, *rows = raw
        width = max(len(header), len(ACCOUNT_COLUMNS))
        header = (list(header) + [""] * width)[:width]
        normalized_rows = [(list(r) + [""] * width)[:width] for r in rows]
        table = pd.DataFrame(normalized_rows, columns=header)
        for col in ACCOUNT_COLUMNS:
            if col not in table.columns:
                table[col] = ""
        table = table[ACCOUNT_COLUMNS].copy()
        table["Nom"] = table["Nom"].astype(str).str.strip()
        before = len(table)
        table = table[table["Nom"].str.lower() != clean.lower()].reset_index(drop=True)
        if len(table) == before:
            return False
        values = [ACCOUNT_COLUMNS] + table[ACCOUNT_COLUMNS].values.tolist()
        ws.clear()
        ws.update(
            values,
            range_name=f"A1:{_a1_column(len(ACCOUNT_COLUMNS))}{max(1, len(values))}",
            value_input_option="USER_ENTERED",
            raw=False,
        )
        _clear_cached_reads()
        return True


def _postprocess_loaded_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Symboles monétaires / espaces insécables (ex. "$1,234.56" ou "1 234 €") avant to_numeric
    _currency_scrub = (
        "Profit",
        "Frais",
        "Prix Entree",
        "Prix Sortie",
        "Quantite",
        "High_Water_Mark_Saisie",
        "High_Water_Mark",
        "Max_Daily_Loss_USD",
    )
    for col in _currency_scrub:
        if col not in df.columns:
            continue
        ser = df[col].astype(str).str.strip()
        ser = ser.str.replace("\u00a0", "", regex=False).str.replace(r"[$€£]", "", regex=True)
        df[col] = ser
    for col in [
        "Prix Entree",
        "Prix Sortie",
        "Quantite",
        "Frais",
        "Profit",
        "Sizing_Score",
        "SL_Score",
        "Revenge_Score",
        "Overtrading_Score",
        "Bias_Score",
        "Execution_Score_Global",
        "High_Water_Mark_Saisie",
        "High_Water_Mark",
        "Profit_Objectif_Pct",
        "Max_Daily_Loss_USD",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [
        "Actif",
        "Type",
        "Setup",
        "Sortie",
        "Session",
        "Etat Mental",
        "Emotion",
        "Biais Jour",
        "Compte",
        "Compte_Type",
        "Image",
    ]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna("")

    df = df.dropna(subset=["Date", "Prix Entree", "Prix Sortie", "Quantite", "Frais", "Profit"])
    return df.sort_values("Date").reset_index(drop=True)


def trades_papa_csv_is_effectively_empty() -> bool:
    """True si ``FILE_PATH`` est absent, quasi vide, ou sans ligne de données utiles."""
    path = _trade_csv_primary_path()
    if not os.path.isfile(path) or os.path.getsize(path) < 5:
        return True
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig")
        if raw.empty:
            return True
        raw = raw.dropna(how="all")
        return raw.empty
    except Exception:
        return True


def filter_trades_par_compte(df: pd.DataFrame, compte_selection: str) -> pd.DataFrame:
    """Filtre la colonne ``Compte`` sans tenir compte de la casse ni des espaces en bord."""
    if df.empty:
        return df.copy()
    sel = str(compte_selection).strip()
    if not sel or sel.casefold() == "tous les comptes":
        return df.copy()
    if "Compte" not in df.columns:
        return df.copy()
    key = sel.casefold()
    col = df["Compte"].astype(str).str.strip().str.casefold()
    return df[col == key].copy()


def load_trades(_refresh_ts: float | None = None) -> pd.DataFrame:
    """Charge les trades depuis la feuille (ou CSV local). ``_refresh_ts`` : valeur ignorée, passée par l'app (ex. ``time.time()``) pour forcer un appel distinct ; ``os.utime`` sur le CSV aide le rafraîchissement disque sous Windows."""
    _ = _refresh_ts
    for path in _trade_csv_paths():
        if os.path.isfile(path):
            try:
                os.utime(path, None)
            except OSError:
                LOGGER.debug("load_trades: utime impossible sur %s", path)
    try:
        with _sheet_lock:
            df = _read_sheet_dataframe()
        return _postprocess_loaded_df(df)
    except Exception as exc:
        local_df = _read_local_trades_dataframe()
        if _is_quota_error(exc):
            LOGGER.warning("load_trades: quota / limite API Google Sheets, lecture locale %s", FILE_PATH)
        else:
            LOGGER.warning("load_trades: échec lecture sheet (%s), tentative locale.", exc)
        if not local_df.empty:
            return _postprocess_loaded_df(local_df)
        if not _is_quota_error(exc):
            _handle_google_exception(exc, "load_trades")
        return _postprocess_loaded_df(local_df)


def save_screenshot(uploaded_file: Any, trade_id: str) -> str:
    """Enregistre une capture dans assets/screenshots/ ; retourne le chemin relatif ou ""."""
    if uploaded_file is None:
        return ""
    safe_id = "".join(c for c in str(trade_id) if c.isalnum() or c in "._-") or "trade"
    orig_name = getattr(uploaded_file, "name", "") or ""
    ext = os.path.splitext(orig_name)[1].lower()
    if ext not in {".png", ".jpg", ".jpeg"}:
        ext = ".png"
    dir_path = os.path.join("assets", "screenshots")
    os.makedirs(dir_path, exist_ok=True)
    filename = f"trade_{safe_id}{ext}"
    full_path = os.path.join(dir_path, filename)
    rel_path = os.path.join("assets", "screenshots", filename).replace("\\", "/")
    data = uploaded_file.getbuffer()
    with open(full_path, "wb") as f:
        f.write(data)
    return rel_path


def resolve_trade_screenshot_path(path: str | None) -> str | None:
    """Retourne un chemin absolu vers le fichier image si présent sur disque, sinon None (pas d'exception)."""
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    p = raw.replace("\\", "/")
    root = os.path.dirname(os.path.abspath(__file__))
    candidates: list[str] = []
    if os.path.isabs(p):
        candidates.append(os.path.normpath(p))
    else:
        candidates.append(os.path.normpath(os.path.join(root, p)))
        candidates.append(os.path.normpath(os.path.join(os.getcwd(), p)))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def save_trade(trade: dict[str, Any]) -> None:
    if not str(trade.get("Session", "")).strip():
        trade["Session"] = get_trading_session()
    row = {col: trade.get(col, _default_for_column(col)) for col in COLUMNS}
    with _sheet_lock:
        df = _read_sheet_dataframe()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = _apply_high_water_mark(df)
        df = df.sort_values("Date", na_position="last").reset_index(drop=True)
        df = df[COLUMNS]
        _write_sheet_dataframe(df)
    _clear_cached_reads()


def append_trades(trades_to_add: pd.DataFrame) -> int:
    if trades_to_add.empty:
        return 0
    with _sheet_lock:
        current = _read_sheet_dataframe()
        merged = pd.concat([current, trades_to_add[COLUMNS]], ignore_index=True)
        merged = _apply_high_water_mark(merged)
        merged = merged.sort_values("Date", na_position="last").reset_index(drop=True)
        merged = merged[COLUMNS]
        _write_sheet_dataframe(merged)
    _clear_cached_reads()
    return int(len(trades_to_add))


def delete_trade_by_position(position: int) -> None:
    """Supprime la ligne a l'index ``position`` dans le journal trie par date (meme ordre que load_trades)."""
    with _sheet_lock:
        df = _read_sheet_dataframe()
        if df.empty:
            return
        df = _postprocess_loaded_df(df)
        if 0 <= position < len(df):
            df = df.drop(index=position).reset_index(drop=True)
            df = _apply_high_water_mark(df)
            df = df.sort_values("Date", na_position="last").reset_index(drop=True)
            df = df[COLUMNS]
            _write_sheet_dataframe(df)
            _clear_cached_reads()


def _normalize_col_name(value: str) -> str:
    s = str(value).strip().lower()
    return "".join(ch for ch in s if ch.isalnum())


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {_normalize_col_name(col): col for col in columns}
    for candidate in candidates:
        key = _normalize_col_name(candidate)
        if key in normalized:
            return normalized[key]
    for col in columns:
        col_norm = _normalize_col_name(col)
        for candidate in candidates:
            cand_norm = _normalize_col_name(candidate)
            if cand_norm and (cand_norm in col_norm or col_norm in cand_norm):
                return col
    return None


def _parse_mixed_datetime(series: pd.Series) -> pd.Series:
    txt = series.astype(str).str.strip()
    dt = pd.to_datetime(txt, errors="coerce", utc=False)
    miss = dt.isna()
    if miss.any():
        dt_alt = pd.to_datetime(txt[miss], errors="coerce", dayfirst=True, utc=False)
        dt.loc[miss] = dt_alt
    miss = dt.isna()
    if miss.any():
        numeric = pd.to_numeric(txt[miss], errors="coerce")
        unix = pd.to_datetime(numeric, unit="s", errors="coerce", utc=False)
        # Si timestamps en millisecondes
        miss_unix = unix.isna()
        if miss_unix.any():
            unix_ms = pd.to_datetime(numeric[miss_unix], unit="ms", errors="coerce", utc=False)
            unix.loc[miss_unix] = unix_ms
        dt.loc[miss] = unix
    return dt


def convert_tradingview_to_mvizion(import_df: pd.DataFrame) -> pd.DataFrame:
    if import_df.empty:
        return pd.DataFrame(columns=COLUMNS)

    cols = [str(c) for c in import_df.columns]
    col_date = find_column(cols, ["date", "time", "timestamp", "open time", "close time", "entry time", "created", "execution time", "datetime"])
    col_date_only = find_column(cols, ["date only", "trade date", "day"])
    col_time_only = find_column(cols, ["time only", "trade time", "hour"])
    col_symbol = find_column(cols, ["symbol", "ticker", "instrument", "asset", "actif", "market", "contract", "instrument name"])
    col_open = find_column(cols, ["open price", "entry price", "open", "buy price", "prix entree", "entry"])
    col_close = find_column(cols, ["close price", "exit price", "close", "sell price", "prix sortie", "exit"])
    col_price = find_column(cols, ["price", "avg price", "fill price", "average price"])
    col_qty = find_column(cols, ["qty", "quantity", "size", "contracts", "quantite", "volume"])
    col_fee = find_column(cols, ["fee", "fees", "commission", "commissions", "frais", "cost"])
    col_profit = find_column(cols, ["profit", "pnl", "net profit", "realized pnl", "result", "pl", "gain", "loss"])
    col_side = find_column(cols, ["type", "side", "direction", "position"])
    col_session = find_column(cols, ["session", "market session"])

    if col_date is None or col_symbol is None:
        return pd.DataFrame(columns=COLUMNS)

    out = pd.DataFrame()
    if col_date:
        out["Date"] = _parse_mixed_datetime(import_df[col_date])
    elif col_date_only and col_time_only:
        combo = import_df[col_date_only].astype(str).str.strip() + " " + import_df[col_time_only].astype(str).str.strip()
        out["Date"] = _parse_mixed_datetime(combo)
    else:
        out["Date"] = pd.NaT
    out["Actif"] = import_df[col_symbol].astype(str).str.strip().str.upper()
    out["Type"] = import_df[col_side].astype(str).str.title() if col_side else "Buy"
    out["Setup"] = ""
    out["Prix Entree"] = pd.to_numeric(import_df[col_open].astype(str).str.replace(",", ".", regex=False).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce") if col_open else None
    out["Prix Sortie"] = pd.to_numeric(import_df[col_close].astype(str).str.replace(",", ".", regex=False).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce") if col_close else None
    if col_price:
        price_series = pd.to_numeric(import_df[col_price].astype(str).str.replace(",", ".", regex=False).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")
        out["Prix Entree"] = out["Prix Entree"].fillna(price_series) if col_open else price_series
        out["Prix Sortie"] = out["Prix Sortie"].fillna(price_series) if col_close else price_series
    out["Prix Entree"] = pd.to_numeric(out["Prix Entree"], errors="coerce").fillna(0.0)
    out["Prix Sortie"] = pd.to_numeric(out["Prix Sortie"], errors="coerce").fillna(0.0)
    out["Quantite"] = pd.to_numeric(import_df[col_qty].astype(str).str.replace(",", ".", regex=False).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce").fillna(1.0) if col_qty else 1.0
    out["Frais"] = pd.to_numeric(import_df[col_fee].astype(str).str.replace(",", ".", regex=False).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce").fillna(0.0) if col_fee else 0.0

    if col_profit:
        out["Profit"] = pd.to_numeric(import_df[col_profit].astype(str).str.replace(",", ".", regex=False).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")
    else:
        out["Profit"] = (out["Prix Sortie"] - out["Prix Entree"]) * out["Quantite"] - out["Frais"]

    out["Sortie"] = "TP"
    out["Session"] = import_df[col_session].astype(str).str.upper() if col_session else get_trading_session()
    out["Etat Mental"] = "Neutre"
    out["Emotion"] = "Calme"
    out["Biais Jour"] = "Haussier"
    out["Compte"] = "Compte 1"
    out["Compte_Type"] = "Eval"
    out["Profit_Objectif_Pct"] = 10.0
    out["Max_Daily_Loss_USD"] = 500.0
    out["Sizing_Score"] = 0.0
    out["SL_Score"] = 0.0
    out["Revenge_Score"] = 0.0
    out["Overtrading_Score"] = 0.0
    out["Bias_Score"] = 0.0
    out["Execution_Score_Global"] = 0.0
    out["High_Water_Mark_Saisie"] = 0.0
    out["High_Water_Mark"] = 0.0
    out["Image"] = ""
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["Date", "Actif", "Profit"])
    out["Profit"] = pd.to_numeric(out["Profit"], errors="coerce")
    out = out.dropna(subset=["Profit"])
    return out[COLUMNS]


def _etat_mental_tvs_points(etat: str) -> float:
    e = str(etat).strip()
    if e in ("Calme", "Concentre", "Confiant"):
        return 10.0
    if e == "Neutre":
        return 5.0
    if e in ("Frustre", "Anxieux", "Fatigue"):
        return 0.0
    return 0.0


def infer_mental_state(
    sizing_score: float,
    sl_score: float,
    revenge_score: float,
    overtrading_score: float,
    bias_score: float,
) -> str:
    # Moyenne simple des 5 notes (0–20), alignée sur le score d'exécution affiché
    score = (sizing_score + sl_score + revenge_score + overtrading_score + bias_score) / 5.0
    if score >= 16:
        return "Calme"
    if score >= 13:
        return "Confiant"
    if score >= 10:
        return "Neutre"
    if score >= 7:
        return "Anxieux"
    return "Frustre"


def get_trading_session() -> str:
    now_paris = datetime.now(ZoneInfo("Europe/Paris"))
    hour = now_paris.hour
    # Fenetres Forex standards approx en heure de Paris
    if 0 <= hour < 8:
        return "ASIA"
    if 8 <= hour < 14:
        return "LONDON"
    if 14 <= hour < 23:
        return "NEW YORK"
    return "OUT"


def _normalize_session_name(value: str) -> str:
    s = str(value).strip().upper()
    if s in {"NEW YORK", "NY"}:
        return "NY"
    if s == "LONDON":
        return "LONDON"
    if s == "ASIA":
        return "ASIA"
    return "OUT"


def trade_timestamps_paris(series: pd.Series) -> pd.Series:
    """Convertit la colonne Date du journal en horodatages Europe/Paris (heure locale)."""
    dt = pd.to_datetime(series, errors="coerce")
    if dt.dt.tz is None:
        return dt.dt.tz_localize("Europe/Paris", ambiguous="infer", nonexistent="shift_forward")
    return dt.dt.tz_convert("Europe/Paris")


def session_insights_paris_heatmap(df: pd.DataFrame) -> tuple[str, str]:
    """
    Retourne (session la plus rentable, jour ouvré le plus risqué) en libellés FR.
    La session suit la colonne Session (agrégée) ; le jour le plus risqué = PnL cumulé le plus faible (Paris).
    """
    if df.empty:
        return "—", "—"
    work = df.copy()
    work["Profit"] = pd.to_numeric(work["Profit"], errors="coerce").fillna(0.0)
    work["SessionNorm"] = work["Session"].map(_normalize_session_name)
    prof = work.groupby("SessionNorm", dropna=False)["Profit"].sum()
    labels_sess = {
        "ASIA": "Asie",
        "LONDON": "Londres",
        "NY": "New York (open US)",
        "OUT": "Hors session",
    }
    if prof.empty:
        best_fr = "—"
    else:
        best_key = str(prof.idxmax())
        best_fr = labels_sess.get(best_key, best_key)

    ts = trade_timestamps_paris(work["Date"])
    w2 = work.copy()
    w2["_ts"] = ts
    w2 = w2.dropna(subset=["_ts"])
    w2["_dow"] = w2["_ts"].dt.dayofweek
    w2 = w2[w2["_dow"] <= 4]
    day_fr = {0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi"}
    w2["_dayname"] = w2["_dow"].map(day_fr)
    by_day = w2.groupby("_dayname", sort=False)["Profit"].sum()
    order = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi"]
    by_day = by_day.reindex([d for d in order if d in by_day.index])
    if by_day.empty:
        worst_fr = "—"
    else:
        worst_fr = str(by_day.idxmin())
    return best_fr, worst_fr


def discipline_score_sur_100(df: pd.DataFrame) -> float:
    """Moyenne des cinq scores d'exécution (0–20 par trade), puis moyenne globale ramenée sur 100 (ex. 15/20 → 75)."""
    cols = ["Sizing_Score", "SL_Score", "Revenge_Score", "Overtrading_Score", "Bias_Score"]
    if df.empty:
        return 0.0
    sub = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    moy_par_trade = sub.mean(axis=1)
    moy = float(moy_par_trade.mean()) if len(moy_par_trade) else 0.0
    return float(max(0.0, min(100.0, moy * 5.0)))


def strategie_conseiller_message(df: pd.DataFrame) -> str:
    """Message court du conseiller stratégique (FR), basé sur les moyennes des critères radar."""
    if df.empty:
        return "Ajoute des trades pour recevoir un conseil adapté à ton profil."
    pairs = [
        ("Contrôle Revenge", "Revenge_Score"),
        ("Sizing", "Sizing_Score"),
        ("Over-trading", "Overtrading_Score"),
        ("Gestion SL", "SL_Score"),
        ("Cohérence biais", "Bias_Score"),
    ]
    cols = [p[1] for p in pairs]
    sub = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    means = {lab: float(sub[col].mean()) for lab, col in pairs}
    if all(v > 15.0 for v in means.values()):
        return "✅ Excellence : Ta discipline est ton plus grand atout. Continue ainsi."
    min_v = min(means.values())
    tol = 1e-6
    lowest = [k for k, v in means.items() if (v - min_v) <= tol]
    if "Contrôle Revenge" in lowest:
        return "⚠️ Alerte : Tes émotions après une perte impactent tes résultats. Respire après un SL."
    if "Sizing" in lowest:
        return "⚠️ Attention : Tes tailles de positions sont irrégulières. Reviens à un risque fixe."
    priority = ["Contrôle Revenge", "Sizing", "Over-trading", "Gestion SL", "Cohérence biais"]
    for lab in priority:
        if lab in lowest:
            return (
                f"⚠️ Point d'attention : {lab} est en retrait par rapport aux autres critères. "
                "Recentre-toi sur ce levier d'amélioration."
            )
    return "⚠️ Continue à suivre tes critères d'exécution jour après jour."


def _dates_libelle_fr(dates: pd.Series) -> list[str]:
    mois = ("janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc.")
    out: list[str] = []
    for ts in dates:
        t = pd.Timestamp(ts)
        out.append(f"{t.day} {mois[t.month - 1]} {t.year}")
    return out


def _setup_display_label(setup: Any) -> str:
    if setup is None:
        return "Non renseigné"
    try:
        if pd.isna(setup):
            return "Non renseigné"
    except (TypeError, ValueError):
        pass
    s = str(setup).strip()
    if not s or s.lower() == "nan":
        return "Non renseigné"
    return s


def performance_par_setup_agregat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrégation par stratégie (colonne Setup) : trades, win rate, profit net, espérance (moyenne du profit par trade).
    """
    cols = ["Setup", "Trades", "Win_Rate_Pct", "Profit_Net", "Esperance"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    w = df.copy()
    if "Setup" not in w.columns:
        w["Setup"] = ""
    w["_sl"] = w["Setup"].apply(_setup_display_label)
    w["Profit"] = pd.to_numeric(w["Profit"], errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    for name, g in w.groupby("_sl", sort=False):
        n = int(len(g))
        wr = float((g["Profit"] > 0).sum() / n * 100.0) if n else 0.0
        net = float(g["Profit"].sum())
        esp = float(g["Profit"].mean()) if n else 0.0
        rows.append(
            {"Setup": name, "Trades": n, "Win_Rate_Pct": wr, "Profit_Net": net, "Esperance": esp}
        )
    out = pd.DataFrame(rows).sort_values("Profit_Net", ascending=False).reset_index(drop=True)
    return out[cols]


def equite_series_dashboard(df: pd.DataFrame) -> dict[str, Any]:
    """
    Données pour la courbe d'équité (PnL cumulé depuis le journal / CSV).
    Retourne dates, équité, ligne des sommets (pour drawdown visuel), agrégats.
    """
    empty: dict[str, Any] = {
        "dates": [],
        "equity": [],
        "peak": [],
        "date_labels": [],
        "profit_total": 0.0,
        "max_drawdown": 0.0,
        "profit_factor": 0.0,
    }
    if df.empty:
        return empty
    work = df.copy().sort_values("Date")
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date"])
    work["Profit"] = pd.to_numeric(work["Profit"], errors="coerce").fillna(0.0)
    equity = work["Profit"].cumsum()
    peak = equity.cummax()
    underwater = peak - equity
    max_dd = float(underwater.max()) if len(underwater) else 0.0

    profits = work["Profit"]
    wins = profits[profits > 0]
    losses = profits[profits < 0]
    gp = float(wins.sum()) if not wins.empty else 0.0
    gl = float(losses.sum())
    if gl != 0:
        pf = gp / abs(gl)
    else:
        pf = gp if gp > 0 else 0.0

    return {
        "dates": work["Date"].dt.to_pydatetime().tolist(),
        "equity": [float(x) for x in equity.tolist()],
        "peak": [float(x) for x in peak.tolist()],
        "date_labels": _dates_libelle_fr(work["Date"]),
        "profit_total": float(profits.sum()),
        "max_drawdown": max_dd,
        "profit_factor": float(pf),
    }


# Heatmap Suivi mensuel (ADN Mat'Sa) — échelle divergente et abscisses courtes FR
PERFORMANCE_HEATMAP_MONTH_LABELS_FR: tuple[str, ...] = (
    "Janv",
    "Févr",
    "Mars",
    "Avr",
    "Mai",
    "Juin",
    "Juil",
    "Août",
    "Sept",
    "Oct",
    "Nov",
    "Déc",
)

PERFORMANCE_HEATMAP_COLORSCALE: list[tuple[float, str]] = [
    (0.0, "#FF4B4B"),
    (0.5, "#161A25"),
    (1.0, "#00FFA3"),
]


def performance_monthly_heatmap_figure(df: pd.DataFrame) -> go.Figure:
    """Heatmap année × mois : profit mensuel ($), échelle ADN (rouge / fond carte / vert néon)."""
    base = "#050505"
    mois_fr = PERFORMANCE_HEATMAP_MONTH_LABELS_FR
    if df.empty:
        return go.Figure().update_layout(
            paper_bgcolor=base,
            plot_bgcolor=base,
            font=dict(color="#E5E7EB"),
            height=420,
            title=dict(
                text="Suivi mensuel — aucun trade pour cette sélection",
                font=dict(size=14, color="#9CA3AF"),
            ),
            margin=dict(l=16, r=16, t=48, b=16),
        )
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date"])
    work["Profit"] = pd.to_numeric(work["Profit"], errors="coerce").fillna(0.0)
    work["Y"] = work["Date"].dt.year.astype(int)
    work["M"] = work["Date"].dt.month.astype(int)
    agg = work.groupby(["Y", "M"], as_index=False)["Profit"].sum()
    years = sorted(agg["Y"].unique().tolist())
    if not years:
        return go.Figure().update_layout(
            paper_bgcolor=base,
            plot_bgcolor=base,
            font=dict(color="#E5E7EB"),
            height=420,
            title=dict(text="Suivi mensuel — dates invalides", font=dict(size=14, color="#9CA3AF")),
            margin=dict(l=16, r=16, t=48, b=16),
        )
    z: list[list[float]] = []
    texts: list[list[str]] = []
    for y in years:
        row_z: list[float] = []
        row_txt: list[str] = []
        for m in range(1, 13):
            sub = agg[(agg["Y"] == y) & (agg["M"] == m)]
            v = float(sub["Profit"].iloc[0]) if not sub.empty else 0.0
            row_z.append(v)
            av = abs(v)
            row_txt.append(f"${v:,.0f}" if av >= 1000 else f"${v:,.2f}")
        z.append(row_z)
        texts.append(row_txt)
    z_flat = [abs(v) for row in z for v in row]
    z_span = max(z_flat) if z_flat else 1.0
    z_span = max(z_span, 1.0)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(mois_fr),
            y=[str(y) for y in years],
            text=texts,
            texttemplate="%{text}",
            textfont=dict(size=11, color="#FFFFFF"),
            colorscale=PERFORMANCE_HEATMAP_COLORSCALE,
            zmid=0.0,
            zmin=-z_span,
            zmax=z_span,
            hovertemplate="Année %{y} · %{x}<br>Profit : %{z:,.2f} $<extra></extra>",
            colorbar=dict(
                title=dict(text="Profit ($)", side="right", font=dict(color="#9CA3AF")),
                tickfont=dict(color="#9CA3AF"),
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text="Heatmap de performance (profit mensuel agrégé)",
            font=dict(color="#F9FAFB", size=15),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor=base,
        plot_bgcolor=base,
        font=dict(color="#E5E7EB"),
        margin=dict(l=16, r=28, t=52, b=88),
        height=max(360, 72 * len(years)),
        xaxis=dict(side="bottom", color="#9CA3AF", tickangle=-32),
        yaxis=dict(title="Année", color="#9CA3AF", autorange="reversed"),
    )
    return fig


def dash_cartes_agregats(df: pd.DataFrame) -> dict[str, float]:
    """
    Agrégats pour la 2e ligne du dashboard (cartes .tv-card) :
    profit total et drawdown max depuis la série d'équité (``equite_series_dashboard``),
    score de discipline global sur 100 (``discipline_score_sur_100``).
    """
    eq = equite_series_dashboard(df)
    return {
        "profit_total": float(eq["profit_total"]),
        "max_drawdown": float(eq["max_drawdown"]),
        "discipline_sur_100": float(discipline_score_sur_100(df)),
    }


def _apply_high_water_mark(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    working = df.copy()
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    working["Profit"] = pd.to_numeric(working["Profit"], errors="coerce").fillna(0.0)
    working["Compte"] = working["Compte"].astype(str).fillna("Compte 1")
    working = working.sort_values(["Compte", "Date"], na_position="last").reset_index(drop=True)
    equity = working.groupby("Compte")["Profit"].cumsum()
    high_water = equity.groupby(working["Compte"]).cummax()
    working["High_Water_Mark"] = high_water
    working = working.sort_index()
    return working


def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    empty = {
        "net_pnl": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_rr_reel": 0.0,
        "sharpe_ratio": 0.0,
        "discipline_score_moyen": 0.0,
        "drawdown_actuel": 0.0,
        "drawdown_pct": 0.0,
        "sorties_tp": 0,
        "sorties_tp_partiel": 0,
        "sorties_sl": 0,
        "sorties_be": 0,
        "profit_par_session": {"ASIA": 0.0, "LONDON": 0.0, "NY": 0.0},
        "winrate_par_session": {"ASIA": 0.0, "LONDON": 0.0, "NY": 0.0},
        "tvs_score": 0.0,
    }
    if df.empty:
        return empty

    profits = df["Profit"].astype(float)
    net_pnl = float(profits.sum())
    win_rate = float((profits > 0).mean() * 100.0)
    wins = profits[profits > 0]
    losses = profits[profits < 0]
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(losses.sum())
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else (gross_profit if gross_profit > 0 else 0.0)

    sort_col = df["Sortie"].astype(str).str.strip()
    sorties_tp = int((sort_col == "TP").sum())
    sorties_tp_partiel = int((sort_col == "TP Partiel").sum())
    sorties_sl = int((sort_col == "SL").sum())
    sorties_be = int((sort_col == "BE").sum())

    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss_abs = float(losses.abs().mean()) if not losses.empty else 0.0
    if avg_loss_abs > 0:
        avg_rr_reel = avg_win / avg_loss_abs
    elif avg_win > 0:
        avg_rr_reel = 99.99
    else:
        avg_rr_reel = 0.0

    # Sharpe Ratio simplifie
    mean_profit = float(np.mean(profits))
    std_profit = float(np.std(profits))
    sharpe_ratio = mean_profit / std_profit if std_profit != 0 else 0.0

    # Score de discipline (moyenne des 5 scores par trade, puis moyenne globale)
    discipline_cols = ["Sizing_Score", "SL_Score", "Revenge_Score", "Overtrading_Score", "Bias_Score"]
    discipline_df = df[discipline_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    discipline_trade = discipline_df.mean(axis=1)
    discipline_score_moyen = float(discipline_trade.mean()) if len(discipline_trade) else 0.0

    perf_brut = (win_rate * 0.5) + (float(profit_factor) * 25.0)
    perf_part = min(60.0, perf_brut)
    mental_pts = df["Etat Mental"].astype(str).map(_etat_mental_tvs_points)
    mental_moy = float(mental_pts.mean()) if len(mental_pts) else 0.0
    psych_part = (mental_moy / 10.0) * 40.0
    tvs_score = float(max(0.0, min(100.0, perf_part + psych_part)))

    # Stats par session
    session_df = df.copy()
    session_df["SessionNorm"] = session_df["Session"].map(_normalize_session_name)
    session_focus = session_df[session_df["SessionNorm"].isin(["ASIA", "LONDON", "NY"])]
    profit_par_session = {"ASIA": 0.0, "LONDON": 0.0, "NY": 0.0}
    winrate_par_session = {"ASIA": 0.0, "LONDON": 0.0, "NY": 0.0}
    if not session_focus.empty:
        for sess in ["ASIA", "LONDON", "NY"]:
            s_df = session_focus[session_focus["SessionNorm"] == sess]
            if not s_df.empty:
                profit_par_session[sess] = float(s_df["Profit"].sum())
                winrate_par_session[sess] = float((s_df["Profit"] > 0).mean() * 100.0)

    # Trailing Drawdown par compte (stocke le sommet atteint par compte)
    dd_df = _apply_high_water_mark(df.copy())
    drawdown_map = {}
    for compte, g in dd_df.groupby("Compte"):
        g_sorted = g.sort_values("Date")
        current_equity = float(g_sorted["Profit"].cumsum().iloc[-1]) if not g_sorted.empty else 0.0
        peak = float(pd.to_numeric(g_sorted["High_Water_Mark"], errors="coerce").fillna(0.0).iloc[-1]) if not g_sorted.empty else 0.0
        dd_abs = max(0.0, peak - current_equity)
        dd_pct = (dd_abs / peak * 100.0) if peak > 0 else 0.0
        drawdown_map[compte] = {"drawdown": dd_abs, "drawdown_pct": dd_pct}

    if drawdown_map:
        drawdown_actuel = max(v["drawdown"] for v in drawdown_map.values())
        drawdown_pct = max(v["drawdown_pct"] for v in drawdown_map.values())
    else:
        drawdown_actuel = 0.0
        drawdown_pct = 0.0

    return {
        "net_pnl": net_pnl,
        "win_rate": win_rate,
        "profit_factor": float(profit_factor),
        "avg_rr_reel": float(avg_rr_reel),
        "sharpe_ratio": float(sharpe_ratio),
        "discipline_score_moyen": float(discipline_score_moyen),
        "drawdown_actuel": float(drawdown_actuel),
        "drawdown_pct": float(drawdown_pct),
        "sorties_tp": sorties_tp,
        "sorties_tp_partiel": sorties_tp_partiel,
        "sorties_sl": sorties_sl,
        "sorties_be": sorties_be,
        "profit_par_session": profit_par_session,
        "winrate_par_session": winrate_par_session,
        "drawdown_par_compte": drawdown_map,
        "tvs_score": tvs_score,
    }


def dash_pnl_par_periode(df: pd.DataFrame, ref: date | None = None) -> dict[str, float]:
    """
    Sommes P&L : aujourd'hui, 7 derniers jours (inclus), mois en cours jusqu'à ``ref``,
    plus périodes précédentes et deltas (vs hier, vs semaine dernière, vs mois civil précédent).

    Clés : ``today``, ``yesterday``, ``week``, ``week_prev``, ``month``, ``month_prev``,
    ``delta_today``, ``delta_week``, ``delta_month``, ``delta_profit_total`` (apport du jour
    au cumul, identique au mouvement « vs veille » du solde).
    """
    ref = ref or date.today()
    z = {
        "today": 0.0,
        "yesterday": 0.0,
        "week": 0.0,
        "week_prev": 0.0,
        "month": 0.0,
        "month_prev": 0.0,
        "delta_today": 0.0,
        "delta_week": 0.0,
        "delta_month": 0.0,
        "delta_profit_total": 0.0,
    }
    if df.empty:
        return z
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date"])
    if work.empty:
        return z
    work["Profit"] = pd.to_numeric(work["Profit"], errors="coerce").fillna(0.0)
    d_cal = work["Date"].dt.normalize().dt.date

    mask_today = d_cal == ref
    y = ref - timedelta(days=1)
    mask_yesterday = d_cal == y

    start_week = ref - timedelta(days=6)
    mask_week = (d_cal >= start_week) & (d_cal <= ref)
    start_prev_week = ref - timedelta(days=13)
    end_prev_week = ref - timedelta(days=7)
    mask_week_prev = (d_cal >= start_prev_week) & (d_cal <= end_prev_week)

    start_month = date(ref.year, ref.month, 1)
    mask_month = (d_cal >= start_month) & (d_cal <= ref)
    if ref.month == 1:
        py, pm = ref.year - 1, 12
    else:
        py, pm = ref.year, ref.month - 1
    last_d = calendar.monthrange(py, pm)[1]
    start_prev_m = date(py, pm, 1)
    end_prev_m = date(py, pm, last_d)
    mask_month_prev = (d_cal >= start_prev_m) & (d_cal <= end_prev_m)

    today = float(work.loc[mask_today, "Profit"].sum())
    yesterday = float(work.loc[mask_yesterday, "Profit"].sum())
    week = float(work.loc[mask_week, "Profit"].sum())
    week_prev = float(work.loc[mask_week_prev, "Profit"].sum())
    month = float(work.loc[mask_month, "Profit"].sum())
    month_prev = float(work.loc[mask_month_prev, "Profit"].sum())

    return {
        "today": today,
        "yesterday": yesterday,
        "week": week,
        "week_prev": week_prev,
        "month": month,
        "month_prev": month_prev,
        "delta_today": today - yesterday,
        "delta_week": week - week_prev,
        "delta_month": month - month_prev,
        "delta_profit_total": today,
    }


def dash_quarter_pnls(df: pd.DataFrame, ref: date | None = None) -> dict[str, float]:
    """P&L du trimestre civil en cours et du trimestre civil précédent (sommes)."""
    ref = ref or date.today()
    z = {"quarter": 0.0, "quarter_prev": 0.0}
    if df.empty:
        return z
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date"])
    if work.empty:
        return z
    work["Profit"] = pd.to_numeric(work["Profit"], errors="coerce").fillna(0.0)
    d_cal = work["Date"].dt.normalize().dt.date

    cq = (ref.month - 1) // 3
    sm = cq * 3 + 1
    start_cq = date(ref.year, sm, 1)
    mask_cq = (d_cal >= start_cq) & (d_cal <= ref)

    prev_q_end = start_cq - timedelta(days=1)
    pqm = 3 * ((prev_q_end.month - 1) // 3) + 1
    prev_q_start = date(prev_q_end.year, pqm, 1)

    mask_pq = (d_cal >= prev_q_start) & (d_cal <= prev_q_end)
    return {
        "quarter": float(work.loc[mask_cq, "Profit"].sum()),
        "quarter_prev": float(work.loc[mask_pq, "Profit"].sum()),
    }


def dash_tradevizion_score_widget(df: pd.DataFrame) -> dict[str, Any]:
    """
    Données pour le widget score central type TradeVizion (score /100, libellé, grille 2×3).
    """
    cm = compute_metrics(df)
    tvs = float(cm.get("tvs_score", 0.0))
    tvs_i = int(round(max(0.0, min(100.0, tvs))))
    if tvs_i >= 75:
        lab = "Expert"
    elif tvs_i >= 60:
        lab = "Confirmé"
    elif tvs_i >= 50:
        lab = "Intermédiaire"
    elif tvs_i >= 25:
        lab = "Débutant+"
    else:
        lab = "Débutant"

    dm = dash_metriques_anneaux(df)
    wr = float(dm["win_rate"])
    pf = float(dm["profit_factor"])
    if not math.isfinite(pf) or pf < 0.0:
        pf = 0.0

    if not df.empty and "Sizing_Score" in df.columns and "SL_Score" in df.columns:
        ss = pd.to_numeric(df["Sizing_Score"], errors="coerce").fillna(10.0)
        sls = pd.to_numeric(df["SL_Score"], errors="coerce").fillna(10.0)
        risk_pts = int(round(float((ss.mean() + sls.mean()) / 2.0)))
        risk_pts = max(0, min(20, risk_pts))
    else:
        risk_pts = 10

    prof_pts = int(round(min(15.0, max(0.0, pf / 2.5 * 15.0))))
    avg_rr = float(dm["avg_rr"])
    if not math.isfinite(avg_rr):
        avg_rr = 0.0
    qual_pts = int(round(min(15.0, max(0.0, min(4.0, avg_rr) / 4.0 * 15.0))))
    if qual_pts < 1 and not df.empty and "Execution_Score_Global" in df.columns:
        exg = pd.to_numeric(df["Execution_Score_Global"], errors="coerce").fillna(0.0)
        qual_pts = int(round(min(15.0, max(0.0, float(exg.mean()) / 20.0 * 15.0))))

    wdp = float(dm["win_day_pct"])
    cons_pts = int(round(min(15.0, max(0.0, wdp / 100.0 * 15.0))))
    if not df.empty and "Bias_Score" in df.columns:
        bs = pd.to_numeric(df["Bias_Score"], errors="coerce").fillna(10.0)
        plan_pts = int(round(min(15.0, max(0.0, float(bs.mean()) / 20.0 * 15.0))))
    else:
        plan_pts = 8

    ix_exec = int(round(dash_indice_execution_0_100(wr, pf)))
    ix_men = int(round(dash_indice_mental_0_100(df)))

    n_tr = int(len(df))
    vol_pts = min(20, max(0, n_tr))
    hint = "Historique insuffisant" if n_tr < 8 else ""

    return {
        "main": tvs_i,
        "label": lab,
        "hint": hint,
        "perf_side": ix_exec,
        "psycho_side": ix_men,
        "volume": (vol_pts, 20),
        "prof_ratio": (prof_pts, 15),
        "consistance": (cons_pts, 15),
        "risk": (risk_pts, 20),
        "qual": (qual_pts, 15),
        "plan": (plan_pts, 15),
    }


def dash_sorties_distribution_counts(df: pd.DataFrame) -> dict[str, int]:
    """Décompte des sorties TP / TP Partiel / SL / BE (libellés du formulaire)."""
    z = {"TP": 0, "TP Partiel": 0, "SL": 0, "BE": 0}
    if df.empty or "Sortie" not in df.columns:
        return z
    sc = df["Sortie"].astype(str).str.strip()
    return {
        "TP": int((sc == "TP").sum()),
        "TP Partiel": int((sc == "TP Partiel").sum()),
        "SL": int((sc == "SL").sum()),
        "BE": int((sc == "BE").sum()),
    }


def dash_emotion_moyenne_0_10(df: pd.DataFrame) -> float:
    """
    Moyenne sur une échelle 0–10 à partir de la colonne ``Emotion`` (texte + emoji).
    Plus la valeur est haute, plus l'état est favorable au score Mat'Sa.
    """
    if df.empty or "Emotion" not in df.columns:
        return 5.0

    def _one(raw: str) -> float:
        s = str(raw).strip().lower()
        if "calme" in s:
            return 10.0
        if "stress" in s:
            return 3.0
        if "euphor" in s:
            return 6.0
        if "peur" in s:
            return 2.0
        return 5.0

    vals = [_one(x) for x in df["Emotion"].astype(str)]
    return float(np.mean(vals)) if vals else 5.0


def matsa_discipline_blend_score(win_rate_pct: float, emotion_moyenne_0_10: float) -> float:
    """Score dynamique : (WinRate × 0,6) + (Moyenne émotion × 4), borné 0–100."""
    s = float(win_rate_pct) * 0.6 + float(emotion_moyenne_0_10) * 4.0
    return float(max(0.0, min(100.0, s)))


def dash_indice_execution_0_100(win_rate_pct: float, profit_factor: float) -> float:
    """Indice d'exécution 0–100 : moitié win rate (cible 60 %), moitié profit factor (cible 2,5)."""
    wr = float(win_rate_pct)
    pf = float(profit_factor)
    if not math.isfinite(pf) or pf < 0.0:
        pf = 0.0
    wr_pt = max(0.0, min(1.0, wr / 60.0)) * 50.0
    pf_pt = max(0.0, min(1.0, pf / 2.5)) * 50.0
    return float(min(100.0, wr_pt + pf_pt))


def dash_indice_mental_0_100(df: pd.DataFrame) -> float:
    """Indice mental 0–100 : uniquement à partir de la colonne ``Emotion`` (échelle slider Mat'Sa)."""
    em = dash_emotion_moyenne_0_10(df)
    return float(max(0.0, min(100.0, em * 10.0)))


def dash_meilleure_serie_gagnants(df: pd.DataFrame) -> int:
    """Plus longue suite de trades gagnants (Profit > 0), ordre chronologique croissant."""
    if df.empty:
        return 0
    work = df.copy()
    work["_d"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.sort_values("_d", ascending=True, kind="mergesort", na_position="first")
    profits = pd.to_numeric(work["Profit"], errors="coerce").fillna(0.0)
    best = cur = 0
    for v in profits:
        if float(v) > 0.0:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def dash_indices_trend_last_n(df: pd.DataFrame, n: int = 10) -> tuple[list[float], list[float]]:
    """
    ``n`` derniers points des indices d'exécution et mental, recalculés après chaque trade
    (ordre chronologique croissant) — pour mini-tendances sous les jauges.
    """
    n = max(3, min(30, int(n)))
    if df.empty:
        z = [0.0] * n
        return z, z
    work = df.copy()
    work["_d"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.sort_values("_d", ascending=True, kind="mergesort", na_position="first")
    L = len(work)
    if L == 0:
        z = [0.0] * n
        return z, z
    start = 0 if L <= n else L - n
    perf: list[float] = []
    mental: list[float] = []
    for i in range(start, L):
        subset = work.iloc[: i + 1]
        dm = dash_metriques_anneaux(subset)
        pfr = float(dm["profit_factor"])
        if not math.isfinite(pfr) or pfr < 0.0:
            pfr = 0.0
        perf.append(dash_indice_execution_0_100(float(dm["win_rate"]), pfr))
        mental.append(dash_indice_mental_0_100(subset))
    while len(perf) < n:
        perf.insert(0, perf[0] if perf else 0.0)
        mental.insert(0, mental[0] if mental else 0.0)
    return perf[-n:], mental[-n:]


def dash_derniers_trades_rows(df: pd.DataFrame, n: int = 5) -> list[pd.Series]:
    """``n`` dernières lignes de trade (les plus récentes), pour les cartes activité."""
    if df.empty:
        return []
    work = df.copy()
    work["_d"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.sort_values("_d", ascending=False, kind="mergesort", na_position="last")
    m = min(int(n), len(work))
    return [work.iloc[i] for i in range(m)]


def matsa_badges_logic(df: pd.DataFrame) -> list[dict[str, str]]:
    """
    Badges simples sur les derniers trades (ordre chronologique inverse = plus récent d'abord).

    - **Série de Feu** : les 3 derniers trades sont tous gagnants (Profit > 0).
    - **Zen** : les 3 derniers trades ont une émotion « Calme ».
    """
    if df.empty or len(df) < 3:
        return []
    work = df.copy()
    work["_d"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.sort_values(["_d"], ascending=[False], kind="mergesort", na_position="last")
    last3 = work.head(3)
    profits = pd.to_numeric(last3["Profit"], errors="coerce").fillna(0.0)
    fire = bool((profits > 0).all())
    zen = False
    if "Emotion" in last3.columns:
        zen = all("calme" in str(e).lower() for e in last3["Emotion"].astype(str))
    out: list[dict[str, str]] = []
    if fire:
        out.append({"id": "serie_feu", "icon": "🔥", "label": "Série de Feu"})
    if zen:
        out.append({"id": "zen", "icon": "🧘", "label": "Zen"})
    return out


def format_date_fr(value: pd.Timestamp | str) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "-"
    return f"{ts.day:02d} {MONTHS_FR[int(ts.month)]} {ts.year}"


def matsa_trade_card(row: pd.Series) -> str:
    """Tuile horizontale trade (Buy/Sell, symbole, date, P&L badge)."""
    typ = str(row.get("Type", "")).strip().upper()
    is_buy = typ == "BUY" or "LONG" in typ or "ACH" in typ
    sym = html.escape(str(row.get("Actif", "")).strip() or "—")
    dt_raw = row.get("Date")
    ts = pd.to_datetime(dt_raw, errors="coerce")
    date_s = html.escape(format_date_fr(ts) if not pd.isna(ts) else str(dt_raw))
    _pnl_raw = pd.to_numeric(row.get("Profit"), errors="coerce")
    pnl = 0.0 if pd.isna(_pnl_raw) else float(_pnl_raw)
    pnl_abs = html.escape(f"${abs(pnl):,.2f}")
    sign = "+" if pnl >= 0 else "−"
    dir_cls = "matsa-trade-card__dir--buy" if is_buy else "matsa-trade-card__dir--sell"
    dir_icon = "▲" if is_buy else "▼"
    badge_cls = "matsa-trade-card__badge--win" if pnl >= 0 else "matsa-trade-card__badge--loss"
    return (
        f'<div class="matsa-dash-card matsa-trade-card" role="group">'
        f'<div class="matsa-trade-card__left">'
        f'<span class="matsa-trade-card__dir {dir_cls}" aria-hidden="true">{dir_icon}</span>'
        f'<span class="matsa-trade-card__sym">{sym}</span>'
        f"</div>"
        f'<div class="matsa-trade-card__center">{date_s}</div>'
        f'<div class="matsa-trade-card__right">'
        f'<span class="matsa-trade-card__badge {badge_cls}">{sign}{pnl_abs}</span>'
        f"</div>"
        "</div>"
    )


def format_month_fr(value: pd.Timestamp | str) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "-"
    return f"{MONTHS_FR[int(ts.month)]} {ts.year}"


def dash_derniers_trades_table(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """``n`` derniers trades : Date, Symbole (Actif), Type, P&L — pour le bloc « Dernières activités »."""
    cols_out = ["Date", "Symbole", "Type", "P&L"]
    if df.empty:
        return pd.DataFrame(columns=cols_out)
    work = df.copy()
    work["_d"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.sort_values(["_d"], ascending=[False], kind="mergesort", na_position="last")
    head = work.head(int(n)).copy()
    sym = head["Actif"].astype(str) if "Actif" in head.columns else pd.Series([""] * len(head))
    typ = head["Type"].astype(str) if "Type" in head.columns else pd.Series([""] * len(head))
    if "Profit" in head.columns:
        pnl = pd.to_numeric(head["Profit"], errors="coerce").fillna(0.0)
    else:
        pnl = pd.Series([0.0] * len(head), dtype=float)

    def _fmt_date(v: object) -> str:
        ts = pd.to_datetime(v, errors="coerce")
        return format_date_fr(ts) if not pd.isna(ts) else str(v)

    out = pd.DataFrame(
        {
            "Date": head["Date"].map(_fmt_date),
            "Symbole": sym,
            "Type": typ,
            "P&L": pnl,
        }
    )
    return out[cols_out].reset_index(drop=True)
