#!/usr/bin/env python3
"""Print a shot from docs/manual/shots as text, row by row, to read its figures.

    .venv/bin/python docs/manual/svgtext.py docs/manual/shots/A/07-filled.svg
"""
import re, sys, html
from collections import defaultdict
s=open(sys.argv[1]).read()
rows=defaultdict(list)
for m in re.finditer(r'<text[^>]*?x="([\d.]+)"[^>]*?y="([\d.]+)"[^>]*>(.*?)</text>', s):
    rows[float(m.group(2))].append((float(m.group(1)), html.unescape(re.sub('<[^>]+>','',m.group(3)))))
for y in sorted(rows):
    line=''.join(t for x,t in sorted(rows[y])).replace('\xa0',' ').rstrip()
    if line.strip(): print(line)
