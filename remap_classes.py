import sys
from pathlib import Path

# stare klasy: 0 player, 1 goalkeeper, 2 referee, 3 ball
# nowe klasy:  0 player, 1 referee, 2 ball
mapping = {0: 0, 1: 0, 2: 1, 3: 2}

labels = Path(sys.argv[1])  # np. dataset/labels
count = 0
for f in labels.rglob("*.txt"):
    out = []
    for line in f.read_text().splitlines():
        parts = line.split()
        if parts:
            parts[0] = str(mapping[int(parts[0])])
            out.append(" ".join(parts))
    f.write_text("\n".join(out) + ("\n" if out else ""))
    count += 1
print(f"Przenumerowano {count} plików")
