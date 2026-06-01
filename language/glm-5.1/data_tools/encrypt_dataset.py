#!/usr/bin/env python3
"""
Encrypt a JSON file (.json) → AES-256-GCM encrypted file (.json.aes).

Usage
-----
    python encrypt_dataset.py input.json                          # prompt for passphrase
    python encrypt_dataset.py input.json -o output.json.aes       # explicit output
    python encrypt_dataset.py input.json -k "my-secret"           # passphrase on command line
    ENCRYPTION_KEY=my-secret python encrypt_dataset.py input.json # via environment

The encrypted format is:  [12-byte nonce][ciphertext || 16-byte GCM tag]

Decryption is handled transparently by ``split_andmapping.load_dataset()``
when ``ENCRYPTION_KEY`` is set in the environment.
"""

import argparse
import getpass
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ── PBKDF2 key derivation (mirrors _derive_key in split_andmapping.py) ──────


def derive_key(key_str: str) -> bytes:
    """Derive a 32-byte AES-256 key from a string passphrase using PBKDF2."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    salt = b"mlperf-glm5.1-aes256-salt"
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=600_000,
    )
    return kdf.derive(key_str.encode("utf-8"))


# ── Encrypt ──────────────────────────────────────────────────────────────────


def encrypt_file(input_path: str, output_path: str, key_str: str) -> None:
    """Read *input_path*, encrypt with AES-256-GCM, write to *output_path*."""
    key = derive_key(key_str)
    aesgcm = AESGCM(key)

    # Generate a random 96-bit nonce
    nonce = os.urandom(12)

    with open(input_path, "rb") as f:
        plaintext = f.read()

    # AESGCM.encrypt returns:  ciphertext || 16-byte tag
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)

    with open(output_path, "wb") as f:
        f.write(nonce)
        f.write(ciphertext)

    orig_mb = len(plaintext) / (1024 * 1024)
    enc_mb = (len(nonce) + len(ciphertext)) / (1024 * 1024)
    print(f"Encrypted:  {input_path}  →  {output_path}")
    print(f"  original : {orig_mb:.2f} MB")
    print(f"  encrypted: {enc_mb:.2f} MB  (nonce + ciphertext + tag)")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encrypt a JSON file with AES-256-GCM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Path to the input JSON file.")
    parser.add_argument(
        "-o", "--output",
        help="Output path (default: <input>.json.aes).",
    )
    parser.add_argument(
        "-k", "--key",
        help="Encryption passphrase.  If omitted, reads from ENCRYPTION_KEY env var or prompts.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    output = args.output or f"{args.input}.aes"

    key_str = args.key or os.environ.get("ENCRYPTION_KEY")
    if not key_str:
        key_str = getpass.getpass("Enter encryption passphrase: ")
        confirm = getpass.getpass("Confirm passphrase: ")
        if key_str != confirm:
            raise SystemExit("Passphrases do not match.")

    encrypt_file(str(input_path), output, key_str)


if __name__ == "__main__":
    main()
