import sys, re, math
# parse the "[ ... ]" embedding vector from main's stdout
def load(path):
    txt = open(path).read()
    m = re.search(r'\[(.*?)\]', txt, re.S)
    if not m: 
        sys.exit("no embedding found in "+path)
    return [float(x) for x in m.group(1).replace(',', ' ').split()]
a = load(sys.argv[1]); b = load(sys.argv[2])
if len(a)!=len(b): sys.exit(f"len mismatch {len(a)} vs {len(b)}")
dot=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b))
cos=dot/(na*nb) if na>0 and nb>0 else 0
maxd=max(abs(x-y) for x,y in zip(a,b))
print(f"cos={cos:.8f} maxabs={maxd:.6f} n={len(a)}")
