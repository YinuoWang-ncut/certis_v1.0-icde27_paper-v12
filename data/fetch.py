#!/usr/bin/env python
"""Fetch the 27 CMS release files this artifact reads, and verify them.

The tables are not redistributed here: the PTP and MUE files carry CPT
descriptions that belong to the American Medical Association, and CMS
distributes them under its own terms. So the artifact ships the manifest ---
URL, byte count, SHA-256 --- and this script, which turns it back into bytes.

    python data/fetch.py            # download what is missing, verify all
    python data/fetch.py --check    # verify only, no network

A file that fails its hash is deleted rather than kept, because a corrupt table
silently changes a count and the whole point of the exercise is the count.

Behind a proxy: set HTTPS_PROXY / HTTP_PROXY, or pass --proxy host:port.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify, do not download")
    ap.add_argument("--proxy", default=None, help="host:port")
    a = ap.parse_args()

    manifest = json.load(open(os.path.join(HERE, "manifest.json"), encoding="utf-8"))
    handlers = []
    if a.proxy:
        handlers.append(urllib.request.ProxyHandler(
            {"http": "http://" + a.proxy, "https": "http://" + a.proxy}))
    opener = urllib.request.build_opener(*handlers)

    bad = 0
    for row in manifest:
        dst = os.path.join(ROOT, row["path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            if a.check:
                print("MISSING  %s" % row["path"])
                bad += 1
                continue
            print("GET      %s" % row["path"])
            req = urllib.request.Request(row["url"], headers={"User-Agent": "curl/8"})
            with opener.open(req, timeout=300) as r, open(dst, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
        size = os.path.getsize(dst)
        digest = sha256(dst)
        if size != row["bytes"] or digest != row["sha256"]:
            print("BAD      %s  size %d (want %d)" % (row["path"], size, row["bytes"]))
            os.remove(dst)
            bad += 1
        else:
            print("OK       %s  %d bytes" % (row["path"], size))

    print("\n%d/%d verified" % (len(manifest) - bad, len(manifest)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
