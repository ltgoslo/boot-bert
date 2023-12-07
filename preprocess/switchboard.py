import sys
from smart_open import open
from normalize import clean


def preprocess(f):
    prev_line = None
    for line in f:
        line = ' '.join(line.strip().split())

        if line.startswith("- "):
            line = line[2:]
        elif line.startswith("-"):
            line = line[1:]

        line = clean(line, minimal=True)

        if len(line) == 0:
            yield ""
            continue

        line = f'"{line}"'

        if prev_line is not None and prev_line == line:
            continue

        yield line
        prev_line = line


input_path = f"../data/babylm_data/babylm_{sys.argv[1]}/switchboard.{sys.argv[1] if sys.argv[1] in ['dev', 'test'] else 'train'}"
output_path = f"../data/processed_{sys.argv[1]}/switchboard.txt"

with open(input_path) as f:
    with open(output_path, 'w') as g:
        for line in preprocess(f):
            g.write(f"{line}\n")
