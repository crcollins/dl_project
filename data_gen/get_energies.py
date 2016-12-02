import sys
import os

base = sys.argv[1]
for fname in sorted(os.listdir(base)):
	path = os.path.join(base, fname)
	with open(path) as f:
		for i, line in enumerate(f):
			if i != 1: continue
			print fname, line.split(';')[0]
