import sys, re
# read bench stdout, emit: label med_ms toks_per_s for key rows
keys=["very-short","short","medium","long","max","bs=1 ","bs=4 ","bs=8 ","bs=16","bs=32"]
rows={}
for line in sys.stdin:
    m=re.match(r'\s*(\S+)\s+tok=\s*(\d+)\s+bs=\s*(\d+).*?med=\s*([\d.]+).*?\|\s*([\d.]+)\s+tok/s\s+([\d.]+)\s+seq/s', line)
    if m:
        label,tok,bs,med,toks,seqs=m.groups()
        rows[label]=(med,toks,seqs)
order=["very-short","short","medium","long","max","bs=1","bs=4","bs=8","bs=16","bs=32"]
for k in order:
    if k in rows:
        med,toks,seqs=rows[k]
        print(f"{k:11s} med={float(med):8.2f}ms  {float(toks):8.0f} tok/s  {float(seqs):7.1f} seq/s")
