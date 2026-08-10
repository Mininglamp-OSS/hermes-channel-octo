"""SSRF-guarded HTTP transport and explicit mutable trust policy."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
import threading
from typing import Any
from urllib.parse import urlparse

import aiohttp
from aiohttp.abc import AbstractResolver, ResolveResult
from aiohttp.resolver import DefaultResolver

_METADATA_HOSTS = frozenset({
    "169.254.169.254",
    "fd00:ec2::254",
    "metadata.google.internal",
    "metadata.goog",
    "metadata",
    "100.100.100.200",
})


def is_private_or_metadata_host(hostname: str) -> bool:
    if not hostname:
        return True
    lowered = hostname.lower()
    if lowered in _METADATA_HOSTS:
        return True
    if lowered.endswith((".local", ".internal", ".localhost")) or lowered == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(lowered.strip("[]"))
    except ValueError:
        return False
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _unconditionally_unsafe_address(address: str) -> bool:
    normalized = address.lower().strip("[]").rstrip(".")
    if normalized in _METADATA_HOSTS:
        return True
    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        return True
    return ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified


def _canonical_literal_ip(host: str) -> str | None:
    normalized = host.lower().strip("[]").rstrip(".")
    try:
        return str(ipaddress.ip_address(normalized))
    except ValueError:
        try:
            return str(ipaddress.ip_address(socket.inet_aton(normalized)))
        except OSError:
            return None


class TransportPolicy:
    """Thread-safe hostname trust authority shared by validation and DNS."""

    def __init__(self, trusted_hosts: set[str] | None = None) -> None:
        self._lock = threading.Lock()
        self._trusted_hosts = {
            host.lower().rstrip(".") for host in (trusted_hosts or set()) if host
        }

    def is_trusted(self, host: str) -> bool:
        normalized = host.lower().strip("[]").rstrip(".")
        with self._lock:
            return normalized in self._trusted_hosts

    def trusted_hosts(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._trusted_hosts)

    def trust_validated_private_host(self, url: str) -> None:
        try:
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower().rstrip(".")
        except (TypeError, ValueError) as exc:
            raise RuntimeError("unsafe presigned upload URL") from exc
        if (
            parsed.scheme not in {"http", "https"}
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or host in _METADATA_HOSTS
        ):
            raise RuntimeError("unsafe presigned upload URL")
        literal = _canonical_literal_ip(host)
        if literal is not None and _unconditionally_unsafe_address(literal):
            raise RuntimeError("unsafe presigned upload URL")
        if os.getenv("OCTO_ALLOW_PRIVATE_HOSTS", "").lower() not in {"1", "true", "yes"}:
            return
        self.trust_host(host)

    def trust_host(self, host: str) -> None:
        normalized = host.lower().strip("[]").rstrip(".")
        if normalized:
            with self._lock:
                self._trusted_hosts.add(normalized)


class _TrustedHostView(set[str]):
    """Compatibility view; mutations still update the policy authority."""

    def __init__(self, policy: TransportPolicy) -> None:
        super().__init__(policy.trusted_hosts())
        self._policy = policy

    def add(self, element: str) -> None:
        self._policy.trust_host(element)
        super().add(element)


class SSRFGuardResolver(AbstractResolver):
    def __init__(
        self,
        *,
        policy: TransportPolicy | None = None,
        trusted_hosts: set[str] | None = None,
    ) -> None:
        super().__init__()
        self.policy = policy or TransportPolicy(trusted_hosts)
        self._delegate = DefaultResolver()

    @property
    def _trusted_hosts(self) -> set[str]:
        return _TrustedHostView(self.policy)

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_INET,
    ) -> list[ResolveResult]:
        normalized = host.lower().rstrip(".")
        if normalized in _METADATA_HOSTS:
            raise OSError(f"unsafe host blocked by SSRF guard: {normalized}")
        trusted = self.policy.is_trusted(normalized)
        if not trusted and is_private_or_metadata_host(normalized):
            raise OSError(f"unsafe host blocked by SSRF guard: {normalized}")
        records = await self._delegate.resolve(host, port, family)
        if not records:
            raise OSError(f"hostname returned no addresses: {normalized}")
        for record in records:
            address = str(record.get("host") or "")
            if _unconditionally_unsafe_address(address) or (
                not trusted and is_private_or_metadata_host(address)
            ):
                raise OSError(f"unsafe address blocked by SSRF guard for {normalized}")
        return records

    async def close(self) -> None:
        await self._delegate.close()


class SSRFGuardConnector(aiohttp.TCPConnector):
    def __init__(
        self,
        *,
        resolver: SSRFGuardResolver,
        policy: TransportPolicy | None = None,
    ) -> None:
        self.policy = policy or resolver.policy
        self._resolver_owner = resolver
        self._owned_resolver = resolver
        self._resolver_closed = False
        self._resolver_close_lock = asyncio.Lock()
        super().__init__(resolver=resolver)

    @property
    def _ssrf_resolver_closed(self) -> bool:
        return self._resolver_closed

    async def _resolve_host(self, host: str, port: int, traces: Any = None) -> list[ResolveResult]:
        normalized = host.lower().strip("[]").rstrip(".")
        trusted = self.policy.is_trusted(normalized)
        literal = _canonical_literal_ip(normalized)
        if normalized in _METADATA_HOSTS:
            raise OSError(f"unsafe host blocked by SSRF guard: {normalized}")
        if literal is not None and (
            _unconditionally_unsafe_address(literal)
            or (not trusted and is_private_or_metadata_host(literal))
        ):
            raise OSError(f"unsafe address blocked by SSRF guard: {normalized}")
        if literal is None and not trusted and is_private_or_metadata_host(normalized):
            raise OSError(f"unsafe host blocked by SSRF guard: {normalized}")
        return await super()._resolve_host(host, port, traces)

    async def close(self, *, abort_ssl: bool = False) -> None:
        try:
            await super().close(abort_ssl=abort_ssl)
        finally:
            async with self._resolver_close_lock:
                if not self._resolver_closed:
                    await self._owned_resolver.close()
                    self._resolver_closed = True


def new_guarded_http_session(*configured_urls: str) -> aiohttp.ClientSession:
    trusted_hosts: set[str] = set()
    if os.getenv("OCTO_ALLOW_PRIVATE_HOSTS", "").lower() in {"1", "true", "yes"}:
        for configured_url in configured_urls:
            if not configured_url:
                continue
            try:
                host = (urlparse(configured_url).hostname or "").lower().rstrip(".")
            except (TypeError, ValueError):
                host = ""
            if host and host not in _METADATA_HOSTS:
                trusted_hosts.add(host)
    policy = TransportPolicy(trusted_hosts)
    resolver = SSRFGuardResolver(policy=policy)
    connector = SSRFGuardConnector(resolver=resolver, policy=policy)
    session = aiohttp.ClientSession(connector=connector)
    session.transport_policy = policy  # type: ignore[attr-defined]
    return session
