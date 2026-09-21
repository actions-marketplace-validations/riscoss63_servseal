"""Poll the probe kernel until it stops, then pull its log and artifacts.

Run it in the background: the kernel installs vLLM and loads three models, so it is
tens of minutes, and there is nothing to do while it is not finished.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from push import KERNEL, fetch, api                                 # noqa: E402

TERMINAL = ("COMPLETE", "ERROR", "CANCEL")


def main():
    a = api()
    t0 = time.time()
    last = None
    while True:
        try:
            s = a.kernels_status(KERNEL)
            st = str(getattr(s, "status", s)).split(".")[-1].upper()
            msg = str(getattr(s, "failure_message", "") or "")
        except Exception as e:
            st, msg = "POLL_ERROR", str(e)
        if st != last:
            print(f"[{time.time() - t0:6.0f}s] {st} {msg}".rstrip(), flush=True)
            last = st
        if any(st.startswith(t) for t in TERMINAL):
            break
        if time.time() - t0 > 5400:
            print("giving up after 90 minutes", flush=True)
            return 1
        time.sleep(30)

    print(f"\nfinal status: {st} {msg}".rstrip(), flush=True)
    try:
        fetch(a)
    except Exception as e:
        print(f"could not fetch output: {e}", flush=True)
        return 1
    return 0 if st.startswith("COMPLETE") else 1


if __name__ == "__main__":
    sys.exit(main())
