import json
import pathlib

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

root = pathlib.Path(__file__).resolve().parent.parent / "spec" / "fixtures" / "template"


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
env.filters["tojson"] = tojson
template = env.from_string((root / "qwen3.jinja").read_text())
cases = json.loads((root / "cases.json").read_text())
(root / "expected").mkdir(exist_ok=True)
for name, case in cases.items():
    text = template.render(
        messages=case["messages"],
        tools=case["tools"] or None,
        add_generation_prompt=True,
        enable_thinking=case["thinking"],
    )
    (root / "expected" / f"{name}.txt").write_text(text)
    print(name, len(text))
