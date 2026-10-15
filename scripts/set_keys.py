"""Ask for your API keys (hidden input) and save them into .env.

    uv run python scripts/set_keys.py

Press Enter on a prompt to keep the current value.
"""

from getpass import getpass
from pathlib import Path

ENV = Path(__file__).resolve().parent.parent / ".env"
KEYS = [
    ("OPENROUTER_API_KEY", "OpenRouter key (sk-or-...)"),
    ("LANGSMITH_API_KEY", "LangSmith key (lsv2_...)"),
]


def main() -> None:
    if not ENV.exists():
        ENV.write_text((ENV.parent / ".env.example").read_text(encoding="utf-8"), encoding="utf-8")
    lines = ENV.read_text(encoding="utf-8").splitlines()
    for name, label in KEYS:
        value = getpass(f"{label} [Enter = keep current]: ").strip()
        if not value:
            continue
        for i, line in enumerate(lines):
            if line.startswith(f"{name}="):
                lines[i] = f"{name}={value}"
                break
        else:
            lines.append(f"{name}={value}")
    ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, _ in KEYS:
        current = next((l.split("=", 1)[1] for l in lines if l.startswith(f"{name}=")), "")
        print(f"{name}: {'set' if current else 'empty'}")


if __name__ == "__main__":
    main()
