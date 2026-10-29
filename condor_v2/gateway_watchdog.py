#!/usr/bin/env python3
"""IB Gateway watchdog (paper DUT155508, port 7497) -- user ask 2026-10-28.

On 2026-10-27 TWS lost its link to IBKR from 14:34 to 16:48 (error 1100). Nothing noticed: the
condor bot went blind, its 15:58 near-money close never ran, and a short 770P was assigned. This
process stays connected to Gateway (read-only, its own client id), watches every layer, and says so
on Telegram the moment something breaks:

  process   IBC's Gateway java (ibcalpha.ibc.IbcGateway) is running
  port      7497 accepts TCP connections
  api       our own API session is up (reconnects itself)
  link      Gateway <-> IBKR servers: error 1100/2110 = lost, 1101/1102 = restored
  data      during RTH, SPY quotes keep ticking (stale > 3 min, after a first tick today)

Escalation while unhealthy (seconds of continuous trouble):
    60   HIGH Telegram with the reason
   180   IBC RECONNECTACCOUNT          (link/data trouble, port still up)
   480   IBC RESTART                   (soft restart, keeps the login)
   900   hard restart: kill Gateway, run C:\\IBC\\StartGateway.bat; again every 20 min after
Gateway not running at all -> start it after 60 s. Recovery is announced with the downtime.
IBC's own nightly restart (23:45 ET) is ignored for 15 minutes.

    python gateway_watchdog.py            # run forever (task IBKR-GatewayWatchdog)
    python gateway_watchdog.py --once     # one health check, print it, exit
"""
import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import condor_v2 as C

ET = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent
LOG = HERE / "gateway_watchdog_log.jsonl"
PORT = 7497
CLIENT_ID = 1690
IBC_CMD_PORT = 7462
START_BAT = r"C:\IBC\StartGateway.bat"
EVERY = 30
STALE_DATA = 180
NIGHTLY = (dtime(23, 40), dtime(23, 59, 59))
NIGHTLY_GRACE = 900
STEPS = ((60, "alert"), (180, "reconnect"), (480, "restart"), (900, "hard_restart"))
HARD_AGAIN = 1200
REMIND = 3600


def log(action, **kw):
    rec = {"time": datetime.now(ET).isoformat(), "action": action, **kw}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{datetime.now(ET):%H:%M:%S}] {action} {kw if kw else ''}", flush=True)


class Telegram:
    """Messages queue while the internet is down (often the very cause) and go out once it's back."""

    def __init__(self, cfg):
        self.cfg, self.queue = cfg, []

    def send(self, text, high=False):
        self.queue.append(("[HIGH] " if high else "") + "GATEWAY: " + text)
        self.flush()

    def flush(self):
        if not (self.cfg.get("telegram_token") and self.cfg.get("telegram_chat_id")):
            self.queue.clear()
            return
        import requests
        while self.queue:
            try:
                r = requests.post(f"https://api.telegram.org/bot{self.cfg['telegram_token']}/sendMessage",
                                  json={"chat_id": self.cfg["telegram_chat_id"], "text": self.queue[0]}, timeout=10)
                if not r.ok:
                    print(f"telegram {r.status_code}", flush=True)
            except Exception:
                return  # still offline: keep the queue
            self.queue.pop(0)


# ------------------------------------------------------------------ probes
def port_open(port=PORT, timeout=3):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def gateway_pids():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='java.exe'\" | "
          "Where-Object { $_.CommandLine -match 'ibcalpha.ibc.IbcGateway' } | ForEach-Object { $_.ProcessId }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                             timeout=30).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return None  # unknown -- don't act on it


def ibc_command(cmd):
    """Send a command to IBC's command server (localhost only). Returns its reply or None."""
    try:
        with socket.create_connection(("127.0.0.1", IBC_CMD_PORT), timeout=5) as s:
            s.sendall((cmd + "\n").encode())
            s.settimeout(10)
            return s.recv(1024).decode(errors="replace").strip()
    except OSError:
        return None


def hard_restart():
    for pid in gateway_pids() or []:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    time.sleep(5)
    subprocess.Popen(["cmd", "/c", "start", "", START_BAT], cwd=r"C:\IBC")


def start_gateway():
    subprocess.Popen(["cmd", "/c", "start", "", START_BAT], cwd=r"C:\IBC")


# ------------------------------------------------------------------ decisions
def in_rth(now):
    return now.weekday() < 5 and dtime(9, 31) <= now.time() <= dtime(15, 59)


def in_nightly(now):
    return NIGHTLY[0] <= now.time() <= NIGHTLY[1] or now.time() <= dtime(0, 5)


def problems(obs):
    """obs: process (bool|None), port, api, link, data (bool|None = not checked)."""
    out = []
    if obs.get("process") is False:
        out.append("Gateway is not running")
    if not obs.get("port"):
        out.append(f"port {PORT} refuses connections")
    elif not obs.get("api"):
        out.append("API session down")
    if obs.get("link") is False:
        out.append("Gateway lost its link to IBKR (1100)")
    if obs.get("data") is False:
        out.append(f"SPY quotes stale > {STALE_DATA // 60} min")
    return out


class Escalator:
    """Pure state machine: feed (now, problem list, process_running) -> list of actions to take."""

    def __init__(self):
        self.since = None
        self.done = set()
        self.last_hard = None
        self.last_alert = None
        self.reasons = []

    def step(self, now, probs, process_running=True, nightly=False):
        acts = []
        if not probs:
            if self.since is not None and "alert" in self.done:  # blips under a minute stay quiet
                acts.append(("recovered", int(now - self.since)))
            self.since, self.done, self.last_hard, self.last_alert, self.reasons = None, set(), None, None, []
            return acts
        if self.since is None:
            self.since = now
        self.reasons = probs
        down = now - self.since
        grace = NIGHTLY_GRACE if nightly else 0
        if process_running is False and down >= 60 + grace and "start" not in self.done:
            self.done.add("start")
            acts.append(("start", int(down)))
        for t, name in STEPS:
            if down >= t + grace and name not in self.done:
                if name in ("reconnect", "restart") and process_running is False:
                    self.done.add(name)  # nothing to talk to
                    continue
                self.done.add(name)
                acts.append((name, int(down)))
                if name == "hard_restart":
                    self.last_hard = now
        if self.last_hard is not None and now - self.last_hard >= HARD_AGAIN:
            self.last_hard = now
            acts.append(("hard_restart", int(down)))
        if "alert" in self.done and self.last_alert is not None and now - self.last_alert >= REMIND:
            acts.append(("remind", int(down)))
        if any(a[0] in ("alert", "remind") for a in acts):
            self.last_alert = now
        return acts


# ------------------------------------------------------------------ runtime
class Watch:
    def __init__(self, cfg):
        from ib_insync import IB
        self.ib = IB()
        self.tg = Telegram(cfg)
        self.link_ok = True
        self.spy = None
        self.last_tick = None
        self.tick_day = None
        self.ib.errorEvent += self.on_error
        self.ib.pendingTickersEvent += self.on_ticks

    def on_error(self, req_id, code, msg, contract=None):
        if code in (1100, 2110):
            if self.link_ok:
                log("link_lost", code=code, msg=msg[:120])
            self.link_ok = False
        elif code in (1101, 1102):
            if not self.link_ok:
                log("link_restored", code=code)
            self.link_ok = True
            if code == 1101 and self.spy is not None:  # data lost: resubscribe
                self.subscribe()

    def on_ticks(self, tickers):
        now = datetime.now(ET)
        self.last_tick, self.tick_day = now, now.date()

    def subscribe(self):
        from ib_insync import Stock
        try:
            c = Stock("SPY", "SMART", "USD")
            self.ib.qualifyContracts(c)
            if self.spy is not None:
                self.ib.cancelMktData(self.spy.contract)
            self.spy = self.ib.reqMktData(c, "", False, False)
        except Exception as e:
            log("subscribe_failed", err=str(e)[:120])

    def ensure_api(self, port_ok):
        if self.ib.isConnected():
            return True
        if not port_ok:
            return False
        try:
            self.ib.connect("127.0.0.1", PORT, clientId=CLIENT_ID, timeout=20, readonly=True)
            acct = self.ib.managedAccounts()[0]
            if not acct.startswith("DU"):
                log("refused_non_paper", account=acct)
                self.ib.disconnect()
                sys.exit(2)
            self.link_ok = True
            self.spy = None
            self.subscribe()
            log("api_connected", account=acct)
            return True
        except SystemExit:
            raise
        except Exception as e:
            try:
                self.ib.disconnect()
            except Exception:
                pass
            log("api_connect_failed", err=str(e)[:120] or type(e).__name__)
            return False

    def observe(self):
        now = datetime.now(ET)
        port_ok = port_open()
        pids = gateway_pids()
        api_ok = self.ensure_api(port_ok)
        data = None
        if api_ok and in_rth(now) and self.tick_day == now.date() and self.last_tick:
            data = (now - self.last_tick).total_seconds() <= STALE_DATA
        return {"process": None if pids is None else bool(pids), "port": port_ok, "api": api_ok,
                "link": self.link_ok if api_ok else None, "data": data}

    def act(self, name, down, reasons):
        mins = f"{down // 60}m{down % 60:02d}s"
        why = "; ".join(reasons)
        log(name, down_s=down, reasons=reasons)
        if name == "alert":
            self.tg.send(f"PROBLEM: {why}. Bots on 7497 (condors, Ashley twin, scalpers) may be blind. "
                         f"Auto-recovery: reconnect at 3 min, restart at 8, hard restart at 15.", high=True)
        elif name == "remind":
            self.tg.send(f"STILL DOWN after {mins}: {why}.", high=True)
        elif name == "reconnect":
            r = ibc_command("RECONNECTACCOUNT")
            self.tg.send(f"down {mins} ({why}) -- sent IBC RECONNECTACCOUNT: {r or 'no reply'}")
        elif name == "restart":
            r = ibc_command("RESTART")
            self.tg.send(f"down {mins} ({why}) -- sent IBC RESTART: {r or 'no reply'}", high=True)
            self.drop_api()
        elif name == "hard_restart":
            self.tg.send(f"down {mins} ({why}) -- HARD RESTART: killing Gateway and relaunching IBC.", high=True)
            self.drop_api()
            hard_restart()
        elif name == "start":
            self.tg.send(f"Gateway was not running ({mins}) -- starting it via IBC.", high=True)
            start_gateway()
        elif name == "recovered":
            self.tg.send(f"RECOVERED after {mins}. Bots that lost their session mid-day may need a check.")

    def drop_api(self):
        try:
            self.ib.disconnect()
        except Exception:
            pass

    def run(self):
        esc = Escalator()
        log("start", port=PORT, client_id=CLIENT_ID)
        while True:
            try:
                obs = self.observe()
                probs = problems(obs)
                for name, down in esc.step(time.time(), probs, obs["process"], in_nightly(datetime.now(ET))):
                    self.act(name, down, esc.reasons or probs)
                self.tg.flush()
            except Exception as e:  # the watchdog must outlive whatever it watches
                log("cycle_error", err=f"{type(e).__name__}: {str(e)[:150]}")
            self.pause()

    def pause(self):
        """ib.sleep() keeps events flowing but raises ConnectionError the moment Gateway's socket
        drops (2026-10-28 test: that killed the first version of this watchdog)."""
        try:
            if self.ib.isConnected():
                self.ib.sleep(EVERY)
                return
        except Exception as e:
            log("api_dropped", err=f"{type(e).__name__}: {str(e)[:120]}")
            self.drop_api()
        time.sleep(EVERY)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    cfg = C.load_config()
    w = Watch(cfg)
    if args.once:
        obs = w.observe()
        w.ib.sleep(3)
        print(json.dumps({**obs, "problems": problems(obs)}))
        w.drop_api()
        return
    w.run()


if __name__ == "__main__":
    main()
