import os
import threading
import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import gspread
import numpy as np
import pandas as pd
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
    "Prix Entree",
    "Prix Sortie",
    "Quantite",
    "Frais",
    "Profit",
    "Sortie",
    "Session",
    "Etat Mental",
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
    "High_Water_Mark",
    "Image",
]
ACCOUNT_COLUMNS = ["Nom", "Objectif_Pct", "Max_Loss_USD", "Initial_Capital"]

# Compatibilite : l'ancienne constante (plus utilisee pour la persistance)
CSV_FILE = "trades_papa.csv"

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


def _write_sheet_dataframe(df: pd.DataFrame) -> None:
    ws = _open_trades_worksheet()
    values = _dataframe_to_sheet_values(df)
    ws.clear()
    last_col = _a1_column(len(COLUMNS))
    last_row = max(1, len(values))
    ws.update(values, range_name=f"A1:{last_col}{last_row}", value_input_option="USER_ENTERED", raw=False)


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
    if col == "Biais Jour":
        return "Haussier"
    if col == "Compte":
        return "Compte 1"
    if col == "Compte_Type":
        return "Eval"
    if col == "Type":
        return "Buy"
    if col == "Session":
        return "London"
    if col == "Image":
        return ""
    return ""


def ensure_csv_exists() -> None:
    """Initialise l'onglet Google Sheet (en-tetes COLUMNS) et migre les colonnes manquantes."""
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


@st.cache_data(ttl=300, show_spinner=False)
def load_accounts_from_sheet() -> pd.DataFrame:
    try:
        with _sheet_lock:
            ensure_csv_exists()
            ws = _open_accounts_worksheet()
            raw = ws.get_all_values()
    except Exception as exc:
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
        ensure_csv_exists()
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
        ensure_csv_exists()
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
        "High_Water_Mark",
        "Profit_Objectif_Pct",
        "Max_Daily_Loss_USD",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["Actif", "Type", "Sortie", "Session", "Etat Mental", "Biais Jour", "Compte", "Compte_Type", "Image"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna("")

    df = df.dropna(subset=["Date", "Prix Entree", "Prix Sortie", "Quantite", "Frais", "Profit"])
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_trades() -> pd.DataFrame:
    try:
        ensure_csv_exists()
        with _sheet_lock:
            df = _read_sheet_dataframe()
        return _postprocess_loaded_df(df)
    except Exception as exc:
        _handle_google_exception(exc, "load_trades")
        return pd.DataFrame(columns=COLUMNS)


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
    # Score global 0-20
    score = (sizing_score + sl_score + (20 - revenge_score) + (20 - overtrading_score) + bias_score) / 5.0
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
        "profit_par_session": profit_par_session,
        "winrate_par_session": winrate_par_session,
        "drawdown_par_compte": drawdown_map,
        "tvs_score": tvs_score,
    }


def format_date_fr(value: pd.Timestamp | str) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "-"
    return f"{ts.day:02d} {MONTHS_FR[int(ts.month)]} {ts.year}"


def format_month_fr(value: pd.Timestamp | str) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "-"
    return f"{MONTHS_FR[int(ts.month)]} {ts.year}"
