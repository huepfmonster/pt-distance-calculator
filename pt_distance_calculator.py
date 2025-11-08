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
    mode,from_stop,to_stop,hint_route_id
    u-bahn,Karlsplatz,Stephansplatz,U1
    tram,Schottentor,Franz-Josefs-Bahnhof,5

Ausgabe je Datei:
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
def load_gtfs_dir(gtfs_dir: str):
    def rd(name, dtype=None):
        path = os.path.join(gtfs_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"GTFS-Datei fehlt: {path}")
        return pd.read_csv(path, dtype=dtype)

    data = {}
    data["stops"] = rd("stops.txt", dtype={"stop_id": str}).fillna("")
    data["stop_times"] = rd("stop_times.txt", dtype={"trip_id": str, "stop_id": str})
    data["trips"] = rd("trips.txt", dtype={"trip_id": str, "route_id": str, "shape_id": str})
    data["routes"] = rd("routes.txt", dtype={"route_id": str})
    try:
        data["shapes"] = rd("shapes.txt", dtype={"shape_id": str})
    except Exception:
        data["shapes"] = pd.DataFrame(columns=[
            "shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence","shape_dist_traveled"
        ])
    return data

# -------------------- Stop-Kandidaten & Trip-Suche --------------------
def candidate_stop_ids(stops: pd.DataFrame, name: str) -> List[str]:
    exact = stops[stops["stop_name"] == name]["stop_id"].tolist()
    if exact:
        return exact
    ci = stops[stops["stop_name"].str.lower() == name.lower()]["stop_id"].tolist()
    if ci:
        return ci
    cont = stops[stops["stop_name"].str.contains(name, case=False, na=False)]["stop_id"].tolist()
    if cont:
        return cont
    raise ValueError(f"Station nicht gefunden: {name}")


def pick_trip_multi(stop_times: pd.DataFrame, trips: pd.DataFrame, routes: pd.DataFrame,
                    a_ids: List[str], b_ids: List[str], hint_route_id: Optional[str] = None) -> Optional[str]:
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
    if hint_route_id:
        mask = cand['route_id'].str.contains(str(hint_route_id), case=False, na=False)
        if mask.any():
            cand = cand[mask]
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
        return None

    shape_id = trips_df.loc[trips_df["trip_id"] == trip_id, "shape_id"].dropna()
    if shape_id.empty:
        return None
    shape_id = shape_id.iloc[0]
    shp = shapes_df[shapes_df["shape_id"] == shape_id].copy()
    if shp.empty:
        return None

    # Cum-Distanzen vorhanden?
    if "shape_dist_traveled" in shp.columns and shp["shape_dist_traveled"].notna().any():
        stops_idx = stops_df.set_index("stop_id")
        ay, ax = stops_idx.loc[stop_id_a, ["stop_lat", "stop_lon"]].astype(float)
        by, bx = stops_idx.loc[stop_id_b, ["stop_lat", "stop_lon"]].astype(float)

        shp["shape_pt_sequence"] = pd.to_numeric(shp["shape_pt_sequence"], errors="coerce").astype("Int64")
        shp = shp.dropna(subset=["shape_pt_sequence"]).sort_values("shape_pt_sequence")
        shp["shape_dist_traveled"] = pd.to_numeric(shp["shape_dist_traveled"], errors="coerce")
        shp = shp.dropna(subset=["shape_dist_traveled"])  # nötig für Distanz

        # nächste Shape-Punkte zu den Stops
        d_a = ((shp["shape_pt_lat"] - ay)**2 + (shp["shape_pt_lon"] - ax)**2).idxmin()
        d_b = ((shp["shape_pt_lat"] - by)**2 + (shp["shape_pt_lon"] - bx)**2).idxmin()
        da = shp.loc[d_a, "shape_dist_traveled"]
        db = shp.loc[d_b, "shape_dist_traveled"]
        if pd.isna(da) or pd.isna(db) or db < da:
            return None

        unit = detect_shape_unit(shp)  # 'm' oder 'km'
        dist_val = float(db - da)
        dist_m = dist_val if unit == 'm' else dist_val * 1000.0
        return dist_m

    # Fallback: polyline summieren (Meter)
    stops_idx = stops_df.set_index("stop_id")
    ay, ax = stops_idx.loc[stop_id_a, ["stop_lat", "stop_lon"]].astype(float)
    by, bx = stops_idx.loc[stop_id_b, ["stop_lat", "stop_lon"]].astype(float)
    shp["shape_pt_sequence"] = pd.to_numeric(shp["shape_pt_sequence"], errors="coerce").astype("Int64")
    shp = shp.dropna(subset=["shape_pt_sequence"]).sort_values("shape_pt_sequence")

    ia = ((shp["shape_pt_lat"] - ay)**2 + (shp["shape_pt_lon"] - ax)**2).idxmin()
    ib = ((shp["shape_pt_lat"] - by)**2 + (shp["shape_pt_lon"] - bx)**2).idxmin()
    i_seq = int(shp.loc[ia, "shape_pt_sequence"]) ; j_seq = int(shp.loc[ib, "shape_pt_sequence"])
    if j_seq < i_seq:
        return None

    part = shp[(shp["shape_pt_sequence"] >= i_seq) & (shp["shape_pt_sequence"] <= j_seq)].copy()
    lat = part["shape_pt_lat"].astype(float).to_numpy()
    lon = part["shape_pt_lon"].astype(float).to_numpy()
    dist = 0.0
    for k in range(1, len(part)):
        dist += haversine(lat[k-1], lon[k-1], lat[k], lon[k])
    return dist

# -------------------- Hauptberechnung --------------------
def compute_legs_km(gtfs_dir: str, legs_csv: str) -> Tuple[pd.DataFrame, float]:
    gtfs = load_gtfs_dir(gtfs_dir)
    stops = gtfs["stops"].copy()
    stop_times = gtfs["stop_times"].copy()
    trips = gtfs["trips"].copy()
    routes = gtfs["routes"].copy()
    shapes = gtfs["shapes"].copy()

    for col in ["stop_lat", "stop_lon"]:
        if col in stops.columns:
            stops[col] = pd.to_numeric(stops[col], errors="coerce")

    legs = pd.read_csv(legs_csv, comment="#")
    out_rows = []
    total_m = 0.0

    for _, row in legs.iterrows():
        mode = str(row.get("mode", ""))
        from_name = str(row["from_stop"]) ; to_name = str(row["to_stop"]) ; hint = row.get("hint_route_id", None)

        # Kandidaten sammeln
        try:
            a_ids = candidate_stop_ids(stops, from_name)
            b_ids = candidate_stop_ids(stops, to_name)
        except Exception as e:
            out_rows.append({"mode": mode, "from": from_name, "to": to_name, "km": np.nan, "note": f"Fehler: {e}"})
            continue

        trip_id = pick_trip_multi(stop_times, trips, routes, a_ids, b_ids, hint)
        if trip_id is None:
            out_rows.append({"mode": mode, "from": from_name, "to": to_name, "km": np.nan, "note": "Kein passender Trip (Richtung) gefunden"})
            continue

        # Im gefundenen Trip die tatsächlich verwendeten Stop-IDs bestimmen
        st = stop_times[stop_times["trip_id"] == trip_id][["stop_id", "stop_sequence"]].copy()
        st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
        st = st.dropna(subset=["stop_sequence"]).sort_values("stop_sequence")
        a_used = st[st["stop_id"].isin(a_ids)].iloc[0]["stop_id"]
        b_used = st[st["stop_id"].isin(b_ids)].iloc[0]["stop_id"]

        dist_m = shape_distance_between(stops, shapes, trips, stop_times, trip_id, a_used, b_used)
        if dist_m is None:
            # Fallback: Luftlinie × Modusfaktor
            ay, ax = stops.set_index("stop_id").loc[a_used, ["stop_lat", "stop_lon"]].astype(float)
            by, bx = stops.set_index("stop_id").loc[b_used, ["stop_lat", "stop_lon"]].astype(float)
            straight = haversine(ay, ax, by, bx)
            factors = {"u-bahn": 1.20, "s-bahn": 1.15, "tram": 1.25, "bus": 1.30}
            factor = factors.get(mode.lower(), 1.25)
            dist_m = straight * factor
            note = "Fallback: Luftlinie×Faktor"
        else:
            note = "Exakt (GTFS shapes)"

        total_m += dist_m
        out_rows.append({"mode": mode, "from": from_name, "to": to_name, "km": round(dist_m / 1000.0, 3), "note": note})

    df = pd.DataFrame(out_rows)
    return df, round(total_m / 1000.0, 3)

# -------------------- Batch/CLI --------------------
def ensure_dirs():
    os.makedirs("./inputs", exist_ok=True)
    os.makedirs("./results", exist_ok=True)


def process_one(gtfs_dir: str, in_path: str, out_dir: str = "./results"):
    base = os.path.basename(in_path)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(out_dir, f"{name}_result.csv")
    df, total_km = compute_legs_km(gtfs_dir, in_path)
    df.to_csv(out_path, index=False)
    print(f"[OK] {base} -> {name}_result.csv | Summe: {total_km:.3f} km")


if __name__ == "__main__":
    import argparse
    ensure_dirs()

    p = argparse.ArgumentParser(description="Kilometer im Wiener Öffi-Netz aus GTFS-Ordner + legs.csv berechnen.")
    p.add_argument("--gtfs_dir", default="./gtfs", help="Pfad zum GTFS-Ordner (mit stops.txt usw.). Default: ./gtfs")
    p.add_argument("--legs", help="Dateiname in ./inputs (z.B. team4.csv). Wenn nicht gesetzt: Batch-Modus über alle CSVs in ./inputs")
    args = p.parse_args()

    inputs_dir = "./inputs"
    results_dir = "./results"

    if args.legs:
        # Einzeldatei: in ./inputs suchen (falls kein Pfad angegeben)
        candidate = args.legs
        if not os.path.isabs(candidate) and os.path.dirname(candidate) == "":
            candidate = os.path.join(inputs_dir, candidate)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Input-Datei nicht gefunden: {candidate}")
        process_one(args.gtfs_dir, candidate, results_dir)
    else:
        # Batch-Modus: alle CSVs in ./inputs*
        files = sorted(glob.glob(os.path.join(inputs_dir, "*.csv")))
        if not files:
            print("Keine CSV-Dateien in ./inputs gefunden. Lege dort z.B. team4.csv ab.")
        for f in files:
            try:
                process_one(args.gtfs_dir, f, results_dir)
            except Exception as e:
                base = os.path.basename(f)
                print(f"[ERR] Fehler bei {base}: {e}")
