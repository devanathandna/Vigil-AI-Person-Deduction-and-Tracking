"""Print a readable movement history from the SQLite log.

  python tools/show_history.py                 # latest session, last 40 rows
  python tools/show_history.py --all-events    # include periodic samples
  python tools/show_history.py --session 20260101_120000 --limit 200
"""
import argparse
import sqlite3


def fmt(v, spec="{:.2f}"):
    return "-" if v is None else spec.format(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/movement.db")
    p.add_argument("--session", default=None)
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--all-events", action="store_true", help="include 'sample' rows")
    a = p.parse_args()

    con = sqlite3.connect(a.db)
    session = a.session or (con.execute("SELECT session FROM movement_events ORDER BY ts DESC LIMIT 1")
                            .fetchone() or [None])[0]
    if session is None:
        print("No data logged yet.")
        return
    where = "session=?" + ("" if a.all_events else " AND event!='sample'")
    rows = con.execute(
        f"SELECT ts_iso,event,target_id,status,cx,cy,dir_image,dir_world,speed_mps,track_conf,dir_conf,camera_moving "
        f"FROM movement_events WHERE {where} ORDER BY ts DESC LIMIT ?", (session, a.limit)).fetchall()[::-1]

    print(f"Session {session}  ({len(rows)} rows shown)")
    print(f"{'time':19} {'event':11} {'id':>3} {'status':9} {'x':>5} {'y':>5} "
          f"{'image dir':22} {'scene dir':22} {'m/s':>5} {'trk':>4} {'dir':>4} cam")
    for r in rows:
        ts, ev, tid, st, cx, cy, di, dw, sp, tc, dc, cm = r
        print(f"{ts:19} {ev:11} {fmt(tid, '{}'):>3} {st or '-':9} {fmt(cx):>5} {fmt(cy):>5} "
              f"{di or '-':22} {dw or '-':22} {fmt(sp):>5} {fmt(tc):>4} {fmt(dc):>4} {'Y' if cm else '-'}")

    print("\nTime share per image direction (from logged samples):")
    for d, n in con.execute("SELECT dir_image, COUNT(*) FROM movement_events "
                            "WHERE session=? AND dir_image IS NOT NULL AND event IN ('sample','change') "
                            "GROUP BY dir_image ORDER BY 2 DESC", (session,)):
        print(f"  {d:24} {n} samples")


if __name__ == "__main__":
    main()
