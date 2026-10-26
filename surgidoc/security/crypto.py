"""Data privacy: AES-256-GCM with a versioned envelope SD1 | key id (4) | nonce (12) | ciphertext+tag.

The associated data binds a ciphertext to its record id, so it cannot be moved to another row. The key id (a hash
prefix) selects the current or previous key, which allows rotation. Keys come from keys.txt and never appear in repr.
"""
import base64
import hashlib
import json
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC, MIN_LEN = b"SD1", 3 + 4 + 12 + 16


class DecryptionError(Exception):
    pass


def generate_key():
    return base64.urlsafe_b64encode(os.urandom(32)).decode()


def _load(key):
    try:
        raw = base64.urlsafe_b64decode(key)
    except (TypeError, ValueError):
        raise ValueError("encryption key must be urlsafe base64 of 32 bytes") from None
    if len(raw) != 32:
        raise ValueError("encryption key must be urlsafe base64 of 32 bytes")
    return hashlib.sha256(raw).digest()[:4], AESGCM(raw)


class Cipher:
    def __init__(self, key, previous=None):
        self.key_id, self._current = _load(key)
        self._ring = {self.key_id: self._current}
        if previous:
            pid, pk = _load(previous)
            self._ring[pid] = pk

    @classmethod
    def from_keys(cls, keys):
        return cls(keys.require("SURGIDOC_ENCRYPTION_KEY"), keys.get("SURGIDOC_ENCRYPTION_KEY_PREVIOUS"))

    def __repr__(self):
        return f"Cipher(key_id={self.key_id.hex()})"

    def encrypt(self, data, aad):
        nonce = os.urandom(12)
        return MAGIC + self.key_id + nonce + self._current.encrypt(nonce, data, aad.encode())

    def decrypt(self, blob, aad):
        if len(blob) < MIN_LEN or not blob.startswith(MAGIC):
            raise DecryptionError("not an SD1 envelope")
        aes = self._ring.get(blob[3:7])
        if aes is None:
            raise DecryptionError("encrypted with an unknown key")
        try:
            return aes.decrypt(blob[7:19], blob[19:], aad.encode())
        except InvalidTag:
            raise DecryptionError("wrong key, wrong record, or data was modified") from None

    def encrypt_json(self, obj, aad):
        return self.encrypt(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode(), aad)

    def decrypt_json(self, blob, aad):
        return json.loads(self.decrypt(blob, aad))
