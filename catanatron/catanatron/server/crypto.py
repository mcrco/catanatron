"""Hashes, signatures and seeds: everything that makes a game auditable.

Every value that crosses the wire as a hash, key, signature or seed is a
lowercase hex string, and every object that is hashed or signed is first
turned into bytes by :func:`canonical`, so a verifier written in any
language reproduces the exact bytes from the parsed JSON.

The scheme (see the server README for the threat model):

* :func:`seed_commitment` -- ``H(server_seed)``, published before any client
  contributes a seed.
* :func:`master_seed` -- ``H(server_seed, client seeds in seat order)``; the
  game's random stream and its board are both derived from it
  (:func:`derive_seed`).
* :func:`state_commitment` -- ``H(canonical(full state) || salt)``, sent with
  every message once the game has started; the salts are revealed at the end.
* :func:`seal` -- a server message: recipient, sequence number, the hash of
  the previous message to that recipient, the state commitment and the
  body, signed with the server's Ed25519 key.
* :func:`action_statement` -- what a client signs to make a move.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from typing import Any, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

#: The ``prev`` of the first message to a recipient.
GENESIS = "0" * 64
SEED_BYTES = 32
SALT_BYTES = 16


def canonical(obj: Any) -> bytes:
    """The one byte encoding of a JSON value that gets hashed or signed.

    Sorted keys, no whitespace, UTF-8, no NaN. Matches a recursive
    sorted-keys ``JSON.stringify`` for the integers and strings the protocol
    uses (it never puts a float in a signed body).
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(obj: Any) -> str:
    return sha256_hex(canonical(obj))


def random_hex(num_bytes: int = SEED_BYTES) -> str:
    return secrets.token_hex(num_bytes)


def is_hex(value: Any, num_bytes: Optional[int] = None) -> bool:
    if not isinstance(value, str) or len(value) % 2 or value != value.lower():
        return False
    try:
        raw = bytes.fromhex(value)
    except ValueError:
        return False
    return num_bytes is None or len(raw) == num_bytes


# ===== keys =====
class SigningKey:
    """An Ed25519 private key that signs canonical JSON."""

    def __init__(self, private_key: Ed25519PrivateKey):
        self._key = private_key
        self.public_hex = (
            private_key.public_key()
            .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            .hex()
        )

    @classmethod
    def generate(cls) -> "SigningKey":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_hex(cls, private_hex: str) -> "SigningKey":
        return cls(Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_hex)))

    def private_hex(self) -> str:
        return self._key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        ).hex()

    @classmethod
    def load_or_create(cls, path: str) -> "SigningKey":
        """The key stored at ``path`` (hex), created there if missing."""
        if os.path.exists(path):
            with open(path) as file:
                return cls.from_hex(file.read().strip())
        key = cls.generate()
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as file:
            file.write(key.private_hex() + "\n")
        return key

    def sign(self, obj: Any) -> str:
        return self._key.sign(canonical(obj)).hex()


def verify_signature(public_hex: str, obj: Any, signature_hex: str) -> bool:
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_hex))
        key.verify(bytes.fromhex(signature_hex), canonical(obj))
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


# ===== seeds =====
def seed_commitment(server_seed: str) -> str:
    return sha256_hex(bytes.fromhex(server_seed))


def master_seed(server_seed: str, client_seeds: list) -> bytes:
    """What every chance event in the game is derived from.

    ``client_seeds`` is in seat order, None for seats that contribute none
    (the server's own bots, whose randomness would be the server's anyway).
    """
    material = {"server_seed": server_seed, "client_seeds": list(client_seeds)}
    return hashlib.sha256(b"catanatron-seed/1" + canonical(material)).digest()


def derive_seed(master: bytes, label: str) -> int:
    """An integer seed for one purpose (``"game"``, ``"map"``)."""
    return int.from_bytes(hashlib.sha256(master + label.encode()).digest(), "big")


def state_commitment(full_state: dict, salt: str) -> str:
    return sha256_hex(canonical(full_state) + bytes.fromhex(salt))


# ===== messages =====
def seal(
    key: SigningKey,
    *,
    to: Optional[str],
    seq: int,
    prev: str,
    commit: Optional[dict],
    body: dict,
) -> dict:
    """A signed server message. ``commit`` is ``{"index", "hash"}`` or None."""
    head = {"to": to, "seq": seq, "prev": prev, "commit": commit, "body": body}
    return {**head, "sig": key.sign(head)}


def head_of(envelope: dict) -> dict:
    return {k: envelope[k] for k in ("to", "seq", "prev", "commit", "body")}


def envelope_hash(envelope: dict) -> str:
    """The chain link: what the next message's ``prev`` must be."""
    return digest(head_of(envelope))


def verify_envelope(server_key: str, envelope: dict) -> bool:
    try:
        return verify_signature(server_key, head_of(envelope), envelope["sig"])
    except KeyError:
        return False


def action_statement(game_id: str, seq: int, decide_hash: str, action: list) -> dict:
    """What a client signs to play ``action`` in answer to the ``decide``
    message number ``seq``, whose hash is ``decide_hash``."""
    return {"game_id": game_id, "seq": seq, "decide": decide_hash, "action": action}
