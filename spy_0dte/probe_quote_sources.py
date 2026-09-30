"""Probe what historical 0DTE SPY option QUOTE data each subscription actually returns.
Prints field names, availability and small samples only -- never credentials."""
import base64
import json
import sys
from pathlib import Path

import requests

cfg = json.loads(Path(sys.argv[1]).read_text())
DAY, EXP = "2025-03-03", "2025-03-03"

print("=== CBOE DataShop / LiveVol: option-and-underlying-quotes ===")
basic = base64.b64encode(f"{cfg['cboe_datashop_client_id']}:{cfg['cboe_datashop_client_secret']}".encode()).decode()
tok = requests.post("https://id.livevol.com/connect/token",
                    headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
                    data={"grant_type": "client_credentials"}, timeout=20)
print("token:", tok.status_code)
if tok.ok:
    H = {"Authorization": f"Bearer {tok.json()['access_token']}"}
    url = "https://api.livevol.com/v1/delayed/allaccess/market/option-and-underlying-quotes"
    for extra in ({}, {"time": "10:00"}, {"time": "10:00:00"}, {"min_time": "10:00:00", "max_time": "10:00:00"}):
        r = requests.get(url, headers=H, params={"symbol": "SPY", "date": DAY, **extra}, timeout=90)
        if not r.ok:
            print(f"  params {extra}: HTTP {r.status_code} {r.text[:120]}")
            continue
        d = r.json()
        opts = d.get("options", [])
        zero = [o for o in opts if str(o.get("expiry", ""))[:10] == EXP]
        top = {k: v for k, v in d.items() if k != "options"}
        print(f"  params {extra}: HTTP 200, {len(opts)} contracts ({len(zero)} expiring that day); top-level: {top}")
        if zero:
            atm = min(zero, key=lambda o: abs(float(o.get("strike", 0)) - float(d.get("underlying_close") or d.get("underlying_last") or 0)))
            print("   one 0DTE record:", json.dumps(atm)[:600])

print("\n=== Polygon / Massive REST: historical option NBBO quotes ===")
key = cfg["polygon_api_key"]
opt = "O:SPY250303C00590000"
for base in ("https://api.polygon.io", "https://api.massive.com"):
    r = requests.get(f"{base}/v3/quotes/{opt}", params={"timestamp.gte": "2025-03-03T15:00:00Z",
                                                          "timestamp.lt": "2025-03-03T15:01:00Z", "limit": 3, "apiKey": key}, timeout=30)
    print(f"  {base} quotes: HTTP {r.status_code}  {r.text[:300]}")
    r = requests.get(f"{base}/v2/aggs/ticker/{opt}/range/1/minute/2025-03-03/2025-03-03",
                     params={"limit": 3, "apiKey": key}, timeout=30)
    print(f"  {base} minute aggs: HTTP {r.status_code}  {r.text[:200]}")
    if r.status_code != 404:
        break

print("\n=== Polygon / Massive S3 flat files ===")
try:
    import boto3
    from botocore.config import Config
    for endpoint in ("https://files.massive.com", "https://files.polygon.io"):
        s3 = boto3.session.Session().client("s3", endpoint_url=endpoint,
                                            aws_access_key_id=cfg["polygon_s3_access_key"],
                                            aws_secret_access_key=cfg["polygon_s3_secret_key"],
                                            config=Config(signature_version="s3v4"))
        try:
            for prefix in ("us_options_opra/", "us_options_opra/quotes_v1/2025/03/", "us_options_opra/minute_aggs_v1/2025/03/"):
                res = s3.list_objects_v2(Bucket="flatfiles", Prefix=prefix, Delimiter="/", MaxKeys=6)
                items = [p["Prefix"] for p in res.get("CommonPrefixes", [])] + [o["Key"] + f" ({o['Size'] / 1e6:.0f} MB)" for o in res.get("Contents", [])]
                print(f"  {endpoint} {prefix}: {items[:6]}")
            break
        except Exception as e:
            print(f"  {endpoint}: {type(e).__name__} {str(e)[:160]}")
except ImportError:
    print("  boto3 not installed in this environment")
