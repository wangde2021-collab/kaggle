#!/usr/bin/env python3
"""Count words in notes/writeup_draft.md (competition limit: <= 2,000).

Counts every whitespace-separated token in the file except HTML-comment
lines (administrative notes, not part of the article). Markdown syntax
(#, *, |, `) is stripped before counting; numbers count as words.
Prints the total and a per-section breakdown.
"""
import re
import sys

PATH = "notes/writeup_draft.md"


def count_words(path):
    total = 0
    sections = {}
    current = "preamble"
    in_comment = False
    for raw in open(path):
        line = raw.rstrip("\n")
        if "<!--" in line:
            in_comment = True
        if in_comment:
            if "-->" in line:
                in_comment = False
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            current = m.group(2).strip()
            sections.setdefault(current, 0)
            line = m.group(2)
        # strip markdown formatting
        line = re.sub(r"[*_`>|]", " ", line)
        line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)
        words = [w for w in line.split() if w.strip()]
        n = len(words)
        total += n
        sections[current] = sections.get(current, 0) + n
    return total, sections


if __name__ == "__main__":
    total, sections = count_words(sys.argv[1] if len(sys.argv) > 1 else PATH)
    print(f"TOTAL WORDS: {total}  (limit 2000, target 1850-1950)")
    for s, n in sections.items():
        print(f"  {n:5d}  {s}")
    sys.exit(0 if total <= 2000 else 1)
