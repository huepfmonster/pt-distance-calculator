#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wien Öffi-Kilometer – v5

Änderungen ggü. v4:
- Standardbetrieb OHNE Parameter: verarbeitet **alle** .csv-Dateien im Unterordner `./inputs/`
  und schreibt Ergebnisse nach `./results/<name>_result.csv`.
- Optional weiterhin: einzelnes File per `--legs <dateiname.csv>` angeben. Es wird in `./inputs/` gesucht
  und das Resultat nach `./results/<name>_result.csv` geschrieben.
- GTFS kommt aus `./gtfs` (Standard, kein Entzippen).
- Einheitenerkennung für `shape_dist_traveled` (m vs. km) bleibt erhalten.

Beispiele:
    # Batch-Modus (alle CSVs in ./inputs)
    python wien_offi_kilometer_v5.py

    # Einzelnes File (wird in ./inputs/ gesucht)
    python wien_offi_kilometer_v5.py --legs team4.csv

CSV-Format (Beispiel):
    line,from_stop,to_stop
    U1,Karlsplatz,Stephansplatz
    5,Schottentor,Franz-Josefs-Bahnhof

(Alternativ akzeptiert das Skript weiterhin Aliasse wie `route`, `linie` oder das alte `hint_route_id` – empfohlen ist jedoch `line`.)

Ausgabe je Datei:
    ./results/<name>_result.csv  (Spalten: original_line, result_line, km, note, from, to)

    ./results/<name>_result.csv  (Spalte "km" in Kilometern, 3 Nachkommastellen)
"""

import os
import glob
import pandas as pd
import numpy as np
from math import radians, sin, cos, asin, sqrt
from typing import List, Optional, Tuple

# -------------------- Geometrie --------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2*asin(sqrt(a))
    return R*c  # meters

# -------------------- GTFS Laden --------------------
def load_gtfs_dir_single(gtfs_dir: str, feed_tag: str):
    """GTFS aus einem Ordner laden und IDs mit feed_tag präfixen,
    damit mehrere Feeds kollisionsfrei zusammengeführt werden können.
    Präfix-Format: "{feed_tag}::<id>".
    """
    def rd(name):
        path = os.path.join(gtfs_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"GTFS-Datei fehlt: {path}")
        return pd.read_csv(path, dtype=str, low_memory=False)

    routes = rd("routes.txt") if os.path.exists(os.path.join(gtfs_dir, "routes.txt")) else pd.DataFrame()
    stops  = rd("stops.txt") if os.path.exists(os.path.join(gtfs_dir, "stops.txt")) else pd.DataFrame()
    trips  = rd("trips.txt") if os.path.exists(os.path.join(gtfs_dir, "trips.txt")) else pd.DataFrame()
    stimes = rd("stop_times.txt") if os.path.exists(os.path.join(gtfs_dir, "stop_times.txt")) else pd.DataFrame()
    shapes = rd("shapes.txt") if os.path.exists(os.path.join(gtfs_dir, "shapes.txt")) else pd.DataFrame()

    # Präfixe anwenden (IDs)
    def pref(col, sfx):
        if col in sfx.columns:
            sfx[col] = sfx[col].astype(str).map(lambda x: f"{feed_tag}::{x}")
        return sfx

    if not routes.empty:
        routes["feed_tag"] = feed_tag
        routes["route_id_raw"] = routes.get("route_id", "")
        routes = pref("route_id", routes)

    if not stops.empty:
        stops["feed_tag"] = feed_tag
        stops["stop_id_raw"] = stops.get("stop_id", "")
        stops = pref("stop_id", stops)

    if not trips.empty:
        trips["feed_tag"] = feed_tag
        trips["trip_id_raw"] = trips.get("trip_id", "")
        trips["route_id_raw"] = trips.get("route_id", "")
        trips["shape_id_raw"] = trips.get("shape_id", "")
        trips = pref("trip_id", trips)
        trips = pref("route_id", trips)
        trips = pref("shape_id", trips)

    if not stimes.empty:
        stimes["feed_tag"] = feed_tag
        stimes = pref("trip_id", stimes)
        stimes = pref("stop_id", stimes)

    if not shapes.empty:
        shapes["feed_tag"] = feed_tag
        shapes["shape_id_raw"] = shapes.get("shape_id", "")
        shapes = pref("shape_id", shapes)

    return {
        "routes": routes,
        "stops": stops,
        "trips": trips,
        "stop_times": stimes,
        "shapes": shapes,
    }


def find_gtfs_dirs(root: str = ".") -> list:
    """Suche alle Unterordner, deren Name mit 'gtfs' beginnt und die minimalen GTFS-Dateien enthalten."""
    dirs = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and name.lower().startswith("gtfs"):
            # mindestens routes/stops/trips/stop_times vorhanden?
            need = ["routes.txt","stops.txt","trips.txt","stop_times.txt"]
            if all(os.path.exists(os.path.join(p, n)) for n in need):
                dirs.append(p)
    return sorted(dirs)


def load_gtfs_dirs(dir_list: list) -> dict:
    """Mehrere Feeds laden, IDs mit feed_tag präfixen und zu einem gemeinsamen Dataset zusammenführen."""
    all_routes, all_stops, all_trips, all_stimes, all_shapes = [], [], [], [], []
    for d in dir_list:
        tag = os.path.basename(os.path.normpath(d))
        dat = load_gtfs_dir_single(d, tag)
        if not dat["routes"].empty: all_routes.append(dat["routes"]) 
        if not dat["stops"].empty:  all_stops.append(dat["stops"]) 
        if not dat["trips"].empty:  all_trips.append(dat["trips"]) 
        if not dat["stop_times"].empty: all_stimes.append(dat["stop_times"]) 
        if not dat["shapes"].empty: all_shapes.append(dat["shapes"]) 

    routes = pd.concat(all_routes, ignore_index=True) if all_routes else pd.DataFrame()
    stops  = pd.concat(all_stops, ignore_index=True)  if all_stops  else pd.DataFrame()
    trips  = pd.concat(all_trips, ignore_index=True)  if all_trips  else pd.DataFrame()
    stimes = pd.concat(all_stimes, ignore_index=True) if all_stimes else pd.DataFrame()
    shapes = pd.concat(all_shapes, ignore_index=True) if all_shapes else pd.DataFrame()

    return {"routes": routes, "stops": stops, "trips": trips, "stop_times": stimes, "shapes": shapes}

# -------------------- Stop-Kandidaten & Trip-Suche --------------------
# Namens-Normalisierung + Varianten (mit/ohne "Wien ")
import unicodedata, re

def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = s.replace("\u00A0", " ")  # NBSP → Space
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _variants_with_wien(raw: str):
    n = _norm_name(raw)
    low = n.casefold()
    variants = [n]
    if not low.startswith("wien "):
        variants.append(f"Wien {n}")
    else:
        variants.append(n[5:].lstrip())
    return variants

# Flexible Hülle um die bestehende candidate_stop_ids-Funktion
# (probiert Name, "Wien "+Name, Name ohne führendes "Wien ")
def candidate_stop_ids_flex(stops: pd.DataFrame, name: str) -> List[str]:
    for v in _variants_with_wien(name):
        try:
            return candidate_stop_ids(stops, v)
        except Exception:
            continue
    # letzter Versuch: contains-Suche auf normalisiertem Namen
    n = _norm_name(name)
    cont = stops[stops["stop_name"].str.contains(n, case=False, na=False)]
    if not cont.empty:
        return cont["stop_id"].tolist()
    raise ValueError(f"Station nicht gefunden: {name}")
def candidate_stop_ids(stops: pd.DataFrame, name: str) -> List[str]:
    # Priorität: exakter Name -> case-insensitive -> contains; bevorzugt child-stops mit parent_station falls vorhanden
    exact = stops[stops["stop_name"] == name]
    if not exact.empty:
        return exact["stop_id"].tolist()
    ci = stops[stops["stop_name"].str.lower() == name.lower()]
    if not ci.empty:
        return ci["stop_id"].tolist()
    contains = stops[stops["stop_name"].str.contains(name, case=False, na=False)]
    if not contains.empty:
        return contains["stop_id"].tolist()
    raise ValueError(f"Station nicht gefunden: {name}")


def pick_trip_multi(stop_times: pd.DataFrame, trips: pd.DataFrame, routes: pd.DataFrame,
                    a_ids: List[str], b_ids: List[str], allowed_route_ids: Optional[set] = None) -> Optional[str]:
    st = stop_times[["trip_id", "stop_id", "stop_sequence"]].copy()
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce").astype("Int64")
    st = st.dropna(subset=["stop_sequence"])  # saubere Datenbasis

    sub = st[st["stop_id"].isin(set(a_ids) | set(b_ids))]
    good_trips = set()
    for tid, grp in sub.groupby("trip_id"):
        ids = set(grp["stop_id"])
        if (set(a_ids) & ids) and (set(b_ids) & ids):
            seq_a = int(grp[grp['stop_id'].isin(a_ids)].iloc[0]['stop_sequence'])
            seq_b = int(grp[grp['stop_id'].isin(b_ids)].iloc[0]['stop_sequence'])
            if seq_a < seq_b:
                good_trips.add(tid)

    if not good_trips:
        return None

    cand = trips[trips['trip_id'].isin(good_trips)].copy()
    if allowed_route_ids:
        cand = cand[cand['route_id'].isin(allowed_route_ids)]
    return cand.iloc[0]['trip_id'] if not cand.empty else None

# -------------------- Einheitencheck & Distanz --------------------
def detect_shape_unit(shp: pd.DataFrame) -> Optional[str]:
    """Gibt 'm' zurück, wenn shape_dist_traveled nach Metern aussieht, sonst 'km'."""
    if "shape_dist_traveled" not in shp.columns or shp["shape_dist_traveled"].isna().all():
        return None
    vals = pd.to_numeric(shp["shape_dist_traveled"], errors="coerce").dropna()
    if vals.empty:
        return None
    maxv = vals.max()
    # Heuristik: Linienformen > 1000 -> i.d.R. Meter (1000 km sind unrealistisch)
    return 'm' if maxv > 1000 else 'km'


def shape_distance_between(stops_df: pd.DataFrame, shapes_df: pd.DataFrame, trips_df: pd.DataFrame,
                           stop_times_df: pd.DataFrame, trip_id: str, stop_id_a: str, stop_id_b: str) -> Optional[float]:
    # 1) Stop-Reihenfolge im Trip prüfen
    st = stop_times_df[stop_times_df["trip_id"] == trip_id].copy()
    st["seq"] = pd.to_numeric(st["stop_sequence"], errors="coerce").astype("Int64")
    st = st.dropna(subset=["seq"]).sort_values("seq")
    if stop_id_a not in set(st["stop_id"]) or stop_id_b not in set(st["stop_id"]):
        return None
    i = st.index[st["stop_id"] == stop_id_a][0]
    j = st.index[st["stop_id"] == stop_id_b][0]
    if st.loc[i, "seq"] == st.loc[j, "seq"]:
        return 0.0
    if st.loc[i, "seq"] > st.loc[j, "seq"]:
        return None  # falsche Richtung

    # 2) Shape holen
    shape_id = trips_df.loc[trips_df["trip_id"] == trip_id, "shape_id"].dropna()
    if shape_id.empty:
        return None
    shape_id = shape_id.iloc[0]
    shp = shapes_df[shapes_df["shape_id"] == shape_id].copy()
    if shp.empty:
        return None

    # 3) Polyline vorbereiten (inkl. nächstliegender Punkte zu den Stops)
    stops_idx = stops_df.set_index("stop_id")
    ay, ax = stops_idx.loc[stop_id_a, ["stop_lat", "stop_lon"]].astype(float)
    by, bx = stops_idx.loc[stop_id_b, ["stop_lat", "stop_lon"]].astype(float)

    shp["shape_pt_sequence"] = pd.to_numeric(shp["shape_pt_sequence"], errors="coerce").astype("Int64")
    shp = shp.dropna(subset=["shape_pt_sequence"]).sort_values("shape_pt_sequence")

    # Indizes der nächstliegenden Shape-Punkte zu A/B
    ia = ((shp["shape_pt_lat"] - ay)**2 + (shp["shape_pt_lon"] - ax)**2).idxmin()
    ib = ((shp["shape_pt_lat"] - by)**2 + (shp["shape_pt_lon"] - bx)**2).idxmin()
    i_seq = int(shp.loc[ia, "shape_pt_sequence"])
    j_seq = int(shp.loc[ib, "shape_pt_sequence"])
    if j_seq < i_seq:
        return None

    # Polyline-Teil zwischen i..j und Distanz in Metern
    part = shp[(shp["shape_pt_sequence"] >= i_seq) & (shp["shape_pt_sequence"] <= j_seq)].copy()
    lat = part["shape_pt_lat"].astype(float).to_numpy()
    lon = part["shape_pt_lon"].astype(float).to_numpy()
    poly_m = 0.0
    for k in range(1, len(part)):
        poly_m += haversine(lat[k-1], lon[k-1], lat[k], lon[k])

    # 4) Falls shape_dist_traveled vorhanden: Differenz nutzen, Einheit automatisch erkennen
    if "shape_dist_traveled" in shp.columns and shp["shape_dist_traveled"].notna().any():
        shp["shape_dist_traveled"] = pd.to_numeric(shp["shape_dist_traveled"], errors="coerce")
        # wir brauchen die Cum-Werte genau an ia/ib
        da = shp.loc[ia, "shape_dist_traveled"] if ia in shp.index else None
        db = shp.loc[ib, "shape_dist_traveled"] if ib in shp.index else None
        if pd.isna(da) or pd.isna(db) or db < da:
            # inkonsistent → bleib bei Polyline
            return poly_m
        raw = float(db - da)
        # Entscheide, ob raw Meter oder Kilometer sind, indem wir mit poly_m vergleichen
        m_err = abs(raw - poly_m)
        km_err = abs(raw * 1000.0 - poly_m)
        return raw if m_err <= km_err else raw * 1000.0

    # 5) Kein Cum-Feld → Polyline-Meter
    return poly_m

# -------------------- Legs-CSV robust einlesen --------------------
def read_legs_csv(path: str) -> pd.DataFrame:
    """Liest Legs-CSV robust ein (Encodings & Trenner), normalisiert Spalten + Unicode,
    fällt zurück, wenn die automatische Trenner-Erkennung versagt, und säubert Leerzeilen.
    Erwartete Pflichtspalten: line (oder route/linie/hint_route_id), from_stop, to_stop.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "iso-8859-15", "mac_roman", "latin1"]
    last_err = None
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, comment="#", sep=None, engine='python', encoding=enc)
            # Fallbacks, falls nur 1 Spalte erkannt wurde oder gar keine Zeilen
            if df is not None and (df.shape[1] == 1 or df.empty):
                try:
                    df = pd.read_csv(path, comment="#", sep=';', engine='python', encoding=enc)
                except Exception:
                    pass
            if df is not None and (df.shape[1] == 1 or df.empty):
                try:
                    df = pd.read_csv(path, comment="#", sep=',', engine='python', encoding=enc)
                except Exception:
                    pass
            if df is not None:
                break
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if df is None:
        raise UnicodeDecodeError("csv-decode", b"", 0, 1, f"Konnte Datei nicht dekodieren; getestete Encodings: {encodings}. Ursprungsfehler: {last_err}")

    # Leerzeilen/Spalten säubern
    df = df.dropna(how='all')

    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*options):
        for o in options:
            if o in cols: return cols[o]
        return None

    col_line = pick("line", "route", "linie", "hint_route_id")
    col_from = pick("from_stop", "from", "start", "von")
    col_to   = pick("to_stop", "to", "ziel", "nach")

    if not col_line or not col_from or not col_to:
        raise ValueError("CSV benötigt Spalten 'line', 'from_stop', 'to_stop' (Aliasse: route/linie/hint_route_id; from/start/von; to/ziel/nach)")

    import unicodedata
    def nfc(x):
        return unicodedata.normalize('NFC', x) if isinstance(x, str) else x

    out = pd.DataFrame({
        "line": df[col_line],
        "from_stop": df[col_from],
        "to_stop": df[col_to],
    })
    for c in ["line", "from_stop", "to_stop"]:
        out[c] = out[c].astype(str).map(nfc).map(lambda s: s.strip())

    # Leere Zeilen nach Normalisierung entfernen
    out = out[~((out['line'] == '') | (out['from_stop'] == '') | (out['to_stop'] == ''))]

    print(f"[INFO] Eingelesen {os.path.basename(path)}: {len(out)} Zeilen, Spalten={list(out.columns)}")
    if out.empty:
        print(f"[WARN] CSV {os.path.basename(path)} hat nach Säuberung keine verwertbaren Zeilen (prüfe Trenner und Header)")
    return out

# -------------------- Hauptberechnung --------------------
def compute_legs_km(gtfs_dirs: list, legs_csv: str) -> Tuple[pd.DataFrame, float]:
    gtfs = load_gtfs_dirs(gtfs_dirs)
    stops = gtfs["stops"].copy()
    stop_times = gtfs["stop_times"].copy()
    trips = gtfs["trips"].copy()
    routes = gtfs["routes"].copy()
    shapes = gtfs["shapes"].copy()

    # Genaue Typkonvertierung nach dem String-Import
    for col in ["stop_lat", "stop_lon"]:
        if col in stops.columns:
            stops[col] = pd.to_numeric(stops[col], errors="coerce")
    # stop_times Sequenzen
    if "stop_sequence" in stop_times.columns:
        stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="coerce")
    # shapes numerisch
    if not shapes.empty:
        if "shape_pt_sequence" in shapes.columns:
            shapes["shape_pt_sequence"] = pd.to_numeric(shapes["shape_pt_sequence"], errors="coerce")
        if "shape_pt_lat" in shapes.columns:
            shapes["shape_pt_lat"] = pd.to_numeric(shapes["shape_pt_lat"], errors="coerce")
        if "shape_pt_lon" in shapes.columns:
            shapes["shape_pt_lon"] = pd.to_numeric(shapes["shape_pt_lon"], errors="coerce")
        if "shape_dist_traveled" in shapes.columns:
            shapes["shape_dist_traveled"] = pd.to_numeric(shapes["shape_dist_traveled"], errors="coerce")

    legs = read_legs_csv(legs_csv)
    out_rows = []
    total_m = 0.0

    # Hilfsfunktion: line-hint -> route_ids
    def route_ids_for_hint(routes_df: pd.DataFrame, hint: str) -> set:
        r = routes_df.copy()
        # vorhandene Spalten sicherstellen
        for col in ["route_id", "route_short_name", "route_long_name"]:
            if col not in r.columns:
                r[col] = ""
        # Normalisierung: Leerzeichen entfernen, casefold
        def norm(s: str) -> str:
            return (s or "").replace(" ", "").replace(" ", "").casefold()

        r["_id"] = r["route_id"].astype(str)
        r["_short"] = r["route_short_name"].astype(str)
        r["_long"] = r["route_long_name"].astype(str)
        h = norm(str(hint).strip())
        # 1) exakter Match short_name / id
        exact = r[(r["_short"].map(norm) == h) | (r["_id"].map(norm) == h)]
        if not exact.empty:
            return set(exact["route_id"].tolist())
        # 2) exakter Match long_name
        exact_long = r[r["_long"].map(norm) == h]
        if not exact_long.empty:
            return set(exact_long["route_id"].tolist())
        # 3) enthält in short/long
        part = r[r["_short"].map(norm).str.contains(h, na=False) | r["_long"].map(norm).str.contains(h, na=False)]
        if not part.empty:
            return set(part["route_id"].tolist())
        # 4) enthält in id (zur Not)
        pid = r[r["_id"].map(norm).str.contains(h, na=False)]
        if not pid.empty:
            return set(pid["route_id"].tolist())
        return set()

    for _, row in legs.iterrows():
        from_name = str(row["from_stop"]).strip()
        to_name = str(row["to_stop"]).strip()
        line_hint = str(row["line"]).strip()

        # 1) Route-Matching aus Linienhinweis
        allowed_routes = route_ids_for_hint(routes, line_hint)
        # Stops auf den/die Feeds der gefundenen Routes einschränken
        feed_tags = set(routes[routes["route_id"].isin(allowed_routes)].get("feed_tag", ""))
        stops_pool = stops[stops.get("feed_tag", "").isin(feed_tags)] if feed_tags else stops

        if not allowed_routes:
            out_rows.append({
                "original_line": line_hint,
                "result_line": "",
                "from": from_name,
                "to": to_name,
                "km": np.nan,
                "note": "Linie nicht im GTFS gefunden"
            })
            continue

        # 2) Stop-Kandidaten finden
        try:
            a_ids = candidate_stop_ids_flex(stops_pool, from_name)
            b_ids = candidate_stop_ids_flex(stops_pool, to_name)
        except Exception as e:
            out_rows.append({
                "original_line": line_hint,
                "result_line": "",
                "from": from_name,
                "to": to_name,
                "km": np.nan,
                "note": f"Fehler: {e}"
            })
            continue

        # 3) Trip wählen (nur erlaubte Routen)
        trip_id = pick_trip_multi(stop_times, trips, routes, a_ids, b_ids, allowed_routes)
        if trip_id is None:
            out_rows.append({
                "original_line": line_hint,
                "result_line": "",
                "from": from_name,
                "to": to_name,
                "km": np.nan,
                "note": "Kein passender Trip (Richtung) gefunden"
            })
            continue

        # 4) Im Trip die tatsächlichen Stop-IDs bestimmen
        st = stop_times[stop_times["trip_id"] == trip_id][["stop_id", "stop_sequence"]].copy()
        st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
        st = st.dropna(subset=["stop_sequence"]).sort_values("stop_sequence")
        a_used = st[st["stop_id"].isin(a_ids)].iloc[0]["stop_id"]
        b_used = st[st["stop_id"].isin(b_ids)].iloc[0]["stop_id"]

        # 5) Distanz berechnen
        dist_m = shape_distance_between(stops, shapes, trips, stop_times, trip_id, a_used, b_used)
        if dist_m is None:
            # Fallback: Luftlinie × generischer Faktor
            ay, ax = stops.set_index("stop_id").loc[a_used, ["stop_lat", "stop_lon"]].astype(float)
            by, bx = stops.set_index("stop_id").loc[b_used, ["stop_lat", "stop_lon"]].astype(float)
            straight = haversine(ay, ax, by, bx)
            factor = 1.22
            dist_m = straight * factor
            note = "Fallback: Luftlinie×Faktor"
        else:
            note = "Exakt (GTFS shapes)"

        total_m += dist_m

        # 6) result_line label aus der verwendeten Route ableiten
        r_id = trips.loc[trips["trip_id"] == trip_id, "route_id"].iloc[0]
        r_row = routes.loc[routes["route_id"] == r_id]
        res_label = ""
        if not r_row.empty:
            sn = str(r_row.iloc[0].get("route_short_name", "") or "").strip()
            ln = str(r_row.iloc[0].get("route_long_name", "") or "").strip()
            res_label = sn if sn else (ln if ln else r_id)

        out_rows.append({
            "original_line": line_hint,
            "result_line": res_label,
            "from": from_name,
            "to": to_name,
            "km": round(dist_m / 1000.0, 3),
            "note": note
        })


    # Abschluss: Ausgabetabelle bauen
    df = pd.DataFrame(out_rows)
    if df.empty:
        print(f"[WARN] Keine Zeilen erzeugt für {os.path.basename(legs_csv)}. Prüfe Spalten (line, from_stop, to_stop) und Linien im GTFS.")
    return df, round(total_m / 1000.0, 3)

# -------------------- Batch/CLI --------------------
def ensure_dirs():
    os.makedirs("./inputs", exist_ok=True)
    os.makedirs("./results", exist_ok=True)


def process_one(gtfs_dirs: list, in_path: str, out_dir: str = "./results"):
    base = os.path.basename(in_path)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(out_dir, f"{name}_result.csv")
    df, total_km = compute_legs_km(gtfs_dirs, in_path)
    df.to_csv(out_path, index=False, encoding='utf-8', lineterminator="\n")
    print(f"[OK] {base} -> {name}_result.csv | Summe: {total_km:.3f} km")


if __name__ == "__main__":
    import argparse
    ensure_dirs()

    p = argparse.ArgumentParser(description="Kilometer im Wiener Öffi-Netz aus GTFS-Ordnern + legs.csv berechnen.")
    p.add_argument("--gtfs_dir", help="Optional: EIN einzelner GTFS-Ordner. Wenn nicht gesetzt, werden ALLE Unterordner verwendet, deren Name mit 'gtfs' beginnt (z.B. ./gtfs_wienerlinien, ./gtfs_oebb).")
    p.add_argument("--legs", help="Dateiname in ./inputs (z.B. team4.csv). Wenn nicht gesetzt: Batch-Modus über alle CSVs in ./inputs")
    args = p.parse_args()

    inputs_dir = "./inputs"
    results_dir = "./results"

    # GTFS-Verzeichnisse bestimmen
    if args.gtfs_dir:
        gtfs_dirs = [args.gtfs_dir]
    else:
        gtfs_dirs = find_gtfs_dirs(".")
        if not gtfs_dirs:
            # Fallback: ./gtfs wenn vorhanden
            if os.path.isdir("./gtfs"):
                gtfs_dirs = ["./gtfs"]
            else:
                raise FileNotFoundError("Keine GTFS-Ordner gefunden. Lege z.B. ./gtfs_wienerlinien oder ./gtfs an.")
    print(f"[INFO] GTFS-Quellen: {', '.join(gtfs_dirs)}")

    if args.legs:
        # Einzeldatei: in ./inputs suchen (falls kein Pfad angegeben)
        candidate = args.legs
        if not os.path.isabs(candidate) and os.path.dirname(candidate) == "":
            candidate = os.path.join(inputs_dir, candidate)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Input-Datei nicht gefunden: {candidate}")
        process_one(gtfs_dirs, candidate, results_dir)
    else:
        # Batch-Modus: alle CSVs in ./inputs*
        files = sorted(glob.glob(os.path.join(inputs_dir, "*.csv")))
        if not files:
            print("Keine CSV-Dateien in ./inputs gefunden. Lege dort z.B. team4.csv ab.")
        for f in files:
            try:
                process_one(gtfs_dirs, f, results_dir)
            except Exception as e:
                base = os.path.basename(f)
                print(f"[ERR] Fehler bei {base}: {e}")
