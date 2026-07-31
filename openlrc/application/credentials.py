"""System-keyring backed credential resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum


class CredentialSource(StrEnum):
    KEYCHAIN = "keychain"
    ENVIRONMENT = "environment"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class ResolvedCredential:
    value: str | None
    source: CredentialSource
    environment_name: str | None = None


ENVIRONMENT_KEYS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY", "OPENROUTER_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_API_KEY",),
    "litellm": ("OPENLRC_LITELLM_API_KEY",),
    "third_party": ("OPENLRC_THIRD_PARTY_API_KEY",),
}


class CredentialStore:
    SERVICE = "openlrc-mac"

    def __init__(self) -> None:
        self.last_error: str | None = None

    def _keyring(self):
        import keyring

        return keyring

    def resolve(self, provider: str) -> ResolvedCredential:
        self.last_error = None
        try:
            value = self._keyring().get_password(self.SERVICE, provider)
        except Exception as exc:
            self.last_error = f"System Keychain is unavailable: {type(exc).__name__}."
            value = None
        if value:
            return ResolvedCredential(value, CredentialSource.KEYCHAIN)
        for name in ENVIRONMENT_KEYS.get(provider, ()):
            value = os.environ.get(name)
            if value:
                return ResolvedCredential(value, CredentialSource.ENVIRONMENT, name)
        return ResolvedCredential(None, CredentialSource.MISSING)

    def set(self, provider: str, value: str) -> None:
        if not value:
            raise ValueError("Credential value cannot be empty.")
        try:
            self._keyring().set_password(self.SERVICE, provider, value)
        except Exception as exc:
            self.last_error = f"System Keychain is unavailable: {type(exc).__name__}."
            raise RuntimeError(
                "The system Keychain is unavailable, so the credential was not saved. Unlock Keychain Access and retry."
            ) from exc
        self.last_error = None

    def delete(self, provider: str) -> None:
        from keyring.errors import PasswordDeleteError

        keyring = self._keyring()
        try:
            keyring.delete_password(self.SERVICE, provider)
        except PasswordDeleteError:
            return
        except Exception as exc:
            self.last_error = f"System Keychain is unavailable: {type(exc).__name__}."
            raise RuntimeError(
                "The system Keychain is unavailable, so the credential was not removed. "
                "Unlock Keychain Access and retry."
            ) from exc
        self.last_error = None
