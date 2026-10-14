"""Signatures and the signed shapes of the protocol.

The scheme (see the server README for the threat model):

* :func:`~catanatron.server.hashing.seed_commitment` -- ``H(server_seed)``,
  published in every ``hello`` before any client contributes.
* :func:`~catanatron.server.hashing.client_seed_commitment` -- each seat's
  commitment to its own seed, sent (with :func:`join_statement` signed as a
  proof of possession of its key) when it joins. Seeds are revealed only
  after the server has published every seat's commitment.
* :func:`~catanatron.server.hashing.master_seed` -- the game's random stream
  and board are derived from the server seed, the revealed client seeds and
  the room's committed config and seats.
* :func:`~catanatron.server.hashing.state_commitment` -- ``H(canonical(full
  state) || salt)``, sent with every message once the game has started; the
  salts are revealed at the end.
* :func:`seal` -- a server message: recipient, sequence number, the hash of
  the previous message to that recipient, the state commitment and the
  body, signed with the server's Ed25519 key.
* :func:`action_statement` -- what a client signs to make a move.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from catanatron.server.hashing import (  # noqa: F401  (re-exported)
    SALT_BYTES,
    SEED_BYTES,
    canonical,
    client_seed_commitment,
    derive_seed,
    digest,
    is_hex,
    master_seed,
    random_hex,
    seed_commitment,
    sha256_hex,
    state_commitment,
)

#: The ``prev`` of the first message to a recipient.
GENESIS = "0" * 64


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


def verify_signature(public_hex: Any, obj: Any, signature_hex: Any) -> bool:
    if not is_hex(public_hex, 32) or not is_hex(signature_hex, 64):
        return False
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_hex))
        key.verify(bytes.fromhex(signature_hex), canonical(obj))
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


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


def verify_envelope(server_key: Any, envelope: Any) -> bool:
    try:
        return verify_signature(server_key, head_of(envelope), envelope["sig"])
    except (KeyError, TypeError):
        return False


def join_statement(
    room_id: str, server_commitment: str, seat: Optional[int], client_commitment: str
) -> dict:
    """What a joining client signs with its seat key: proof that it holds
    the key, bound to the room, the server's seed commitment it saw, the
    seat it asked for and its own seed commitment."""
    return {
        "domain": "catanatron-join/1",
        "room_id": room_id,
        "seed_commitment": server_commitment,
        "seat": seat,
        "client_commitment": client_commitment,
    }


def action_statement(
    room_id: str, game_id: str, index: int, seq: int, decide_hash: str, action: list
) -> dict:
    """What a client signs to play ``action`` as the game's action number
    ``index``, in answer to the ``decide`` message number ``seq`` (whose hash
    is ``decide_hash``)."""
    return {
        "room_id": room_id,
        "game_id": game_id,
        "index": index,
        "seq": seq,
        "decide": decide_hash,
        "action": action,
    }
