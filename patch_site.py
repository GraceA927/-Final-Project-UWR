import re
p = "Site_Setup.py"
s = open(p).read()

# 1. IPTp: cap at 1 with a warning instead of aborting
pat = re.compile(r'raise ValueError\(f"\{IPTP_FILE\}: year \{r\[.year.\]\} coverage rises with dose "\s*'
                 r'f"\(cumulative coverage must not increase\)\."\)')
new = '''print(f"  WARNING {IPTP_FILE}: year {r['year']} dose coverage {c:.3f} exceeds previous {prev:.3f}; conditional coverage capped at 1.")
                cond.append(1.0)'''
if len(pat.findall(s)) == 1:
    s = pat.sub(lambda m: new, s)
    print("IPTp: patched")
else:
    print("IPTp: NOT patched (pattern not found once)")

# 2. only pass event= when the bednet function accepts it
if "def _bednet_accepts_event" not in s and s.count("    if event:\n") == 1 and s.count("def add_itn_triggered(") == 1:
    s = s.replace("    if event:\n", "    if event and _bednet_accepts_event():\n")
    helper = '''def _bednet_accepts_event():
    names = ("event_name", "broadcast_event", "received_event",
             "bednet_received_event", "intervention_distributed_event")
    params = inspect.signature(add_triggered_usage_dependent_bednet).parameters
    return any(n in params for n in names)


'''
    s = s.replace("def add_itn_triggered(", helper + "def add_itn_triggered(")
    print("event: patched")
else:
    print("event: NOT patched (see grep below)")

open(p, "w").write(s)
