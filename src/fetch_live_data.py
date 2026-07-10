"""
Fetches live data from the AAMS API and writes it to local disk using the
EXACT SAME schema as your historical data/raw/{date}/*.json files and
catalogue.json — so build_dataset.py and labeling.py run completely
unchanged on live data, instead of a separate parallel in-memory pipeline.

Why the catalogue merge matters: the live GET /ml/bearings response often
doesn't include bearing geometry (innerRacePass/outerRacePass/rollElementPass/
cageRotation), even though your bulk catalogue.json does. Those fields are
needed both as direct model features and to derive shaft_frequency-based
fault frequencies for the envelope_amp features. build_merged_catalogue()
fills in whatever the live API is missing from your local catalogue.json,
keyed by bearingLocationId, and reports any bearings that are missing
geometry in BOTH sources.

Fetching strategy: one API call per (bearing, axis), parallelized with a
thread pool. Uses a hard wall-clock deadline (concurrent.futures.wait with
timeout) rather than waiting indefinitely for every request — a handful of
requests can hang on a flaky connection even with reasonable per-request
timeouts, and one stuck request shouldn't block an entire day's run. Whatever
completed before the deadline is kept; stragglers are logged and skipped.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.api_client import AAMSClient

GEOMETRY_COLS = ["innerRacePass", "outerRacePass", "rollElementPass", "cageRotation",
                  "bearingNumber", "manufacturerName", "type"]


def _save_raw_response(day_dir: Path, bearing_id: str, b: dict, resp: dict, target_date: str) -> bool:
    """
    Stores the API response as close to verbatim as possible - just the
    bearing metadata + whatever packets came back, no per-axis splitting or
    field renaming. packets can (and normally will) span multiple axes in a
    single response; build_dataset.py already reads the axis per-packet, not
    per-file, so this needs no other changes downstream.
    """
    if not resp or not resp.get("packets"):
        return False
    doc = {
        "bearingLocationId": bearing_id,
        "source": b.get("source"),
        "date": target_date,
        "count": resp.get("count", len(resp.get("packets", []))),
        "status": b.get("statusName"),
        "customerName": b.get("customerName"),
        "machineName": b.get("machineName"),
        "rpm": b.get("machineRpm"),
        "measuringType": b.get("bearingMeasuringType"),
        "packets": resp["packets"],
    }
    out_path = day_dir / f"{bearing_id}.json"
    with open(out_path, "w") as f:
        json.dump(doc, f)
    return True


def _fetch_one(client: AAMSClient, bearing: dict, target_date: str):
    """One API call per bearing (no axis param -> API returns ALL axes, per
    the documented default). Previously this looped per-axis, needlessly
    multiplying request count by up to 3x."""
    bearing_id = bearing["bearingLocationId"]
    try:
        resp = client.get_raw(bearing_id, date=target_date)
    except Exception as e:
        return bearing_id, None, str(e)
    return bearing_id, resp, None


def save_live_waveforms(target_date: str, save_root: Path,
                         measuring_types=None, max_workers: int = 40, max_total_seconds: int = 1800):
    """
    Pulls every bearing's raw waveform for target_date from the live API and
    writes one JSON file per (bearing, axis) to save_root/{target_date}/,
    in the same shape as your historical files.

    measuring_types: None (default) fetches ALL types in a single paginated
    catalogue walk (per the API's documented default when measuringType is
    omitted) — much faster than looping once per type. Pass an explicit
    tuple like ("sensor",) only if you want to filter to specific types.

    max_workers: number of concurrent API requests (network-bound — this is
    the main lever for speed). The client's connection pool is sized to
    match this, so threads don't block waiting for a free pooled connection.

    max_total_seconds: hard wall-clock deadline for the whole batch (default
    30 min). If reached, whatever's already been saved is kept and the run
    moves on instead of hanging on a few stuck requests. Set to None to wait
    indefinitely (not recommended for large fleets).

    Returns the list of live bearing catalogue records fetched along the way
    (so we don't need a second API round-trip to build the catalogue).
    """
    client = AAMSClient(pool_maxsize=max_workers)
    client.login()

    print("Fetching bearing catalogue from API...")
    if measuring_types:
        bearings = []
        for mt in measuring_types:
            bearings.extend(client.list_bearings(measuring_type=mt))
    else:
        bearings = client.list_bearings()  # no filter -> all types, one paginated walk
    print(f"Total bearings: {len(bearings)}")

    day_dir = save_root / target_date
    day_dir.mkdir(parents=True, exist_ok=True)

    bearing_by_id = {b["bearingLocationId"]: b for b in bearings}
    total_tasks = len(bearings)
    print(f"Fetching {total_tasks} bearing waveforms (all axes per call) with {max_workers} parallel workers, "
          f"deadline {max_total_seconds}s..." if max_total_seconds else
          f"Fetching {total_tasks} bearing waveforms (all axes per call) with {max_workers} parallel workers, no deadline...")

    files_written = 0
    completed = 0
    start = time.time()

    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {
            pool.submit(_fetch_one, client, b, target_date): b["bearingLocationId"]
            for b in bearings
        }

        done, not_done = wait(futures.keys(), timeout=max_total_seconds)

        for fut in done:
            bearing_id = futures[fut]
            completed += 1
            if completed % 50 == 0 or completed == total_tasks:
                elapsed = time.time() - start
                print(f"Progress: {completed}/{total_tasks} bearings done "
                      f"({files_written} files saved so far, {elapsed:.0f}s elapsed)")
            try:
                bid, resp, err = fut.result()
                if err:
                    print(f"  [WARN] {bid}: {err}")
                    continue
                b = bearing_by_id[bid]
                if _save_raw_response(day_dir, bid, b, resp, target_date):
                    files_written += 1
            except Exception as e:
                print(f"  [WARN] task failed for bearing={bearing_id}: {e}")

        if not_done:
            stuck = [futures[f] for f in not_done]
            print(f"  [WARN] Deadline ({max_total_seconds}s) reached with {len(not_done)}/{total_tasks} "
                  f"requests still hanging — giving up on them and moving on. "
                  f"Stuck bearings (sample): {stuck[:10]}")
    except KeyboardInterrupt:
        print(f"Interrupted by user after {completed}/{total_tasks} tasks. "
              f"{files_written} files saved so far — exiting cleanly.")
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    print(f"Wrote {files_written} waveform files -> {day_dir}")
    return bearings


def build_merged_catalogue(live_bearings: list, local_catalogue_path: Path, save_path: Path) -> pd.DataFrame:
    """
    Merges live bearing records with local catalogue.json, filling in
    GEOMETRY_COLS from the local file wherever the live API left them missing.
    Live values (machineRpm, statusName, etc.) always take priority when
    present. Writes the merged result to save_path and returns it as a
    DataFrame. Also prints any bearings still missing geometry after the
    merge (present in neither source, or only partially).
    """
    live_df = pd.DataFrame(live_bearings)

    if local_catalogue_path.exists():
        with open(local_catalogue_path) as f:
            local_records = json.load(f)
        local_df = pd.DataFrame(local_records)
    else:
        print(f"  [WARN] local catalogue not found at {local_catalogue_path} — "
              f"cannot fill in missing geometry, relying on live API only")
        local_df = pd.DataFrame(columns=["bearingLocationId"] + GEOMETRY_COLS)

    local_geom_cols = [c for c in GEOMETRY_COLS if c in local_df.columns]
    local_geom = (
        local_df[["bearingLocationId"] + local_geom_cols]
        .drop_duplicates(subset="bearingLocationId", keep="last")
        if local_geom_cols else pd.DataFrame(columns=["bearingLocationId"])
    )

    merged = live_df.merge(local_geom, on="bearingLocationId", how="left", suffixes=("", "_local"))
    for col in GEOMETRY_COLS:
        local_col = f"{col}_local"
        if col in merged.columns and local_col in merged.columns:
            merged[col] = merged[col].fillna(merged[local_col])
            merged.drop(columns=[local_col], inplace=True)
        elif local_col in merged.columns:
            merged[col] = merged[local_col]
            merged.drop(columns=[local_col], inplace=True)
        elif col not in merged.columns:
            merged[col] = pd.NA

    present_geom_cols = [c for c in GEOMETRY_COLS if c in merged.columns]
    merged["geometry_known"] = ~merged[present_geom_cols].isna().any(axis=1)

    still_missing = merged[~merged["geometry_known"]]
    if len(still_missing):
        print(f"  [WARN] {len(still_missing)}/{len(merged)} bearings still missing geometry "
              f"after merging with local catalogue (not in live API OR local catalogue.json):")
        for bid in still_missing["bearingLocationId"].head(20).tolist():
            print(f"           {bid}")
        if len(still_missing) > 20:
            print(f"           ... and {len(still_missing) - 20} more")
        print("         These rows will fall back to training-set median values for the missing "
              "fields rather than crashing — but their fault-frequency-derived features "
              "(envelope_amp) will be less reliable. Consider adding these bearings to your "
              "local catalogue.json if you have their geometry.")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_json(save_path, orient="records", indent=2)
    print(f"Wrote merged catalogue ({len(merged)} bearings) -> {save_path}")

    return merged
