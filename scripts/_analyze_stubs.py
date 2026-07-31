import ast
import pathlib

src = pathlib.Path('src/aether')
results = []
for f in sorted(src.rglob('*.py')):
    try:
        tree = ast.parse(f.read_text(encoding='utf-8', errors='ignore'))
    except Exception:
        continue
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    funcs   = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    stubs = 0
    for fn in funcs:
        body = fn.body
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            stubs += 1
        elif len(body) <= 2:
            non_doc = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(getattr(s, 'value', None), ast.Constant))]
            if len(non_doc) == 0:
                stubs += 1
    results.append((str(f.relative_to(src)), len(classes), len(funcs), stubs, f.stat().st_size))

header = "Module".ljust(55) + "Cls".rjust(4) + "Fns".rjust(5) + "Stubs".rjust(6) + "Bytes".rjust(8)
print(header)
print('-' * 80)
for name, c, fn, s, b in results:
    if b > 500:
        stub_pct = int(s / fn * 100) if fn else 0
        if stub_pct > 70:
            flag = ' <<MOSTLY STUB'
        elif stub_pct > 30:
            flag = ' <<PARTIAL'
        else:
            flag = ''
        line = name.ljust(55) + str(c).rjust(4) + str(fn).rjust(5) + str(s).rjust(6) + str(b).rjust(8) + flag
        print(line)
