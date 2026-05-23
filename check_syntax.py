import ast
import os
import sys

errors = []

for root, dirs, files in os.walk('agent'):
    for fname in files:
        if not fname.endswith('.py'):
            continue
        fpath = os.path.join(root, fname)
        try:
            with open(fpath) as f:
                source = f.read()
            ast.parse(source, filename=fpath)
            print(f'  OK  {fpath}')
        except SyntaxError as e:
            errors.append((fpath, str(e)))
            print(f'  ERR {fpath}: {e}')

if errors:
    print(f'\n{len(errors)} SYNTAX ERROR(S) FOUND')
    sys.exit(1)
else:
    print(f'\nAll {sum(1 for r,_,fs in os.walk("agent") for f in fs if f.endswith(".py"))} files pass syntax check.')
