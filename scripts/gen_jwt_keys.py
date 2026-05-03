"""Generate an RS256 keypair and append JWT_ lines to .env in UTF-8.

Usage:
    python scripts/gen_jwt_keys.py           # appends to .env
    python scripts/gen_jwt_keys.py --print   # prints to stdout only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def generate() -> tuple[str, str]:
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    # Encode newlines so value fits on a single .env line
    return (
        private_pem.replace("\n", "\\n"),
        public_pem.replace("\n", "\\n"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", help="Print to stdout only, don't write file")
    args = parser.parse_args()

    private_inline, public_inline = generate()
    lines = [
        f'JWT_PRIVATE_KEY="{private_inline}"\n',
        f'JWT_PUBLIC_KEY="{public_inline}"\n',
    ]

    if args.print:
        sys.stdout.writelines(lines)
        return

    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        sys.exit(f"No .env file found at {env_path}. Copy .env.example first.")

    # Remove any existing JWT_PRIVATE_KEY / JWT_PUBLIC_KEY lines first
    existing = env_path.read_text(encoding="utf-8", errors="replace")
    cleaned = "\n".join(
        line for line in existing.splitlines()
        if not line.startswith("JWT_PRIVATE_KEY=") and not line.startswith("JWT_PUBLIC_KEY=")
    ).rstrip() + "\n"

    with env_path.open("w", encoding="utf-8") as f:
        f.write(cleaned)
        f.writelines(lines)

    print(f"JWT keys written to {env_path}")


if __name__ == "__main__":
    main()
