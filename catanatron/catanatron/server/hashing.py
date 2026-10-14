"""Canonical JSON, hashes, seeds and commitments: the standard library only.

Kept apart from :mod:`catanatron.server.crypto` (signatures, which need the
``cryptography`` package) so that building a game from a config and a seed
works without it.

Every value that crosses the wire as a hash or seed is a lowercase hex
string, and every object that is hashed or signed is first turned into bytes
by :func:`canonical`, so a verifier in any language reproduces the exact
bytes from the parsed JSON.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from typing import Any, Optional

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


#: The largest integer every client represents exactly (JavaScript's
#: Number.MAX_SAFE_INTEGER): the protocol's numbers stay within it.
MAX_SAFE_INTEGER = 2**53 - 1


def numbers_ok(value: Any) -> bool:
    """Whether every number in ``value`` is an integer every client reads
    alike (no floats, nothing beyond 2**53)."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return abs(value) <= MAX_SAFE_INTEGER
    if isinstance(value, list):
        return all(numbers_ok(v) for v in value)
    if isinstance(value, dict):
        return all(numbers_ok(v) for v in value.values())
    return False


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


def seed_commitment(server_seed: str) -> str:
    """The server's commitment, published before anyone contributes."""
    return sha256_hex(bytes.fromhex(server_seed))


def client_seed_commitment(room_id: str, public_key: str, client_seed: str) -> str:
    """A seat's commitment to its seed, bound to the room and its key so
    that nobody can copy another seat's commitment and later open it."""
    return digest(
        {
            "domain": "catanatron-client-seed/1",
            "room_id": room_id,
            "public_key": public_key,
            "client_seed": client_seed,
        }
    )


def master_seed(server_seed: str, client_seeds: list, binding: Any = None) -> bytes:
    """What every chance event in the game is derived from.

    ``client_seeds`` is in seat order, None for seats that contribute none
    (the server's own bots). ``binding`` is folded in as well: the room, its
    config and its seats, as committed before the seeds were revealed.
    """
    material = {"server_seed": server_seed, "client_seeds": list(client_seeds)}
    if binding is not None:
        material["binding"] = binding
    return hashlib.sha256(b"catanatron-seed/1" + canonical(material)).digest()


def derive_seed(master: bytes, label: str) -> int:
    """An integer seed for one purpose (``"game"``, ``"map"``)."""
    return int.from_bytes(hashlib.sha256(master + label.encode()).digest(), "big")


def state_commitment(full_state: dict, salt: str) -> str:
    return sha256_hex(canonical(full_state) + bytes.fromhex(salt))
