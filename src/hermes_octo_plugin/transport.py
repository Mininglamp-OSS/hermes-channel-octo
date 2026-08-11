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

from yarl import URL

_METADATA_HOSTS = frozenset({
    "169.254.169.254",
    "fd00:ec2::254",
    "metadata.google.internal",
    "metadata.goog",
    "metadata",
    "100.100.100.200",
})


def private_hosts_enabled() -> bool:
    """Return whether the operator explicitly trusts configured private origins."""
    return os.getenv("OCTO_ALLOW_PRIVATE_HOSTS", "").lower() in {"1", "true", "yes"}



def is_private_or_metadata_host(hostname: str) -> bool:
    normalized = _canonical_trust_host(hostname)
    if not normalized:
        return True
    if normalized in _METADATA_HOSTS:
        return True
    if normalized.endswith((".local", ".internal", ".localhost")) or normalized == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if str(ip) in _METADATA_HOSTS:
        return True
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
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if str(ip) in _METADATA_HOSTS:
        return True
    if ip.is_loopback:
        return False
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


def canonical_url_host(url: str) -> str | None:
    """Return yarl-normalized, literal-canonicalized host for one URL."""
    try:
        host = URL(url).raw_host
    except (TypeError, UnicodeError, ValueError):
        return None
    if not host:
        return None
    normalized = host.lower().strip("[]").rstrip(".")
    return _canonical_literal_ip(normalized) or normalized


def _canonical_trust_host(host: str) -> str:
    normalized = host.strip("[]").rstrip(".")
    try:
        return canonical_url_host(
            str(URL.build(scheme="http", host=normalized))
        ) or normalized.lower()
    except (TypeError, UnicodeError, ValueError):
        return ""


TransportOrigin = tuple[str, str, int]
TransportEndpoint = tuple[str, int]
_DEFAULT_ORIGIN_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def _canonical_origin(url: str) -> TransportOrigin | None:
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        host = canonical_url_host(url)
        port = parsed.port
    except (TypeError, ValueError):
        return None
    if (
        scheme not in _DEFAULT_ORIGIN_PORTS
        or not host
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return scheme, host, port if port is not None else _DEFAULT_ORIGIN_PORTS[scheme]


def _safe_private_trust_origin(url: str) -> TransportOrigin | None:
    origin = _canonical_origin(url)
    if origin is None:
        return None
    _, host, _ = origin
    if host in _METADATA_HOSTS:
        return None
    literal = _canonical_literal_ip(host)
    if literal is not None and _unconditionally_unsafe_address(literal):
        return None
    return origin


class TransportPolicy:
    """Thread-safe origin trust authority shared by validation and DNS."""

    def __init__(self, trusted_origins: set[str] | None = None) -> None:
        self._lock = threading.Lock()
        self._trusted_download_origins = {
            origin
            for value in (trusted_origins or set())
            if (origin := _safe_private_trust_origin(value)) is not None
        }
        self._trusted_connection_endpoints = {
            (host, port) for _, host, port in self._trusted_download_origins
        }

    def is_download_url_trusted(self, url: str) -> bool:
        origin = _canonical_origin(url)
        if origin is None:
            return False
        with self._lock:
            return origin in self._trusted_download_origins

    def is_download_endpoint_trusted(self, url: str) -> bool:
        """Return whether *url* reuses a configured download host and port."""
        origin = _canonical_origin(url)
        if origin is None:
            return False
        _, host, port = origin
        with self._lock:
            return (host, port) in self._trusted_connection_endpoints

    def is_connection_trusted(self, host: str, port: int) -> bool:
        endpoint = (_canonical_trust_host(host), port)
        with self._lock:
            return endpoint in self._trusted_connection_endpoints

    def trusted_download_origins(self) -> frozenset[TransportOrigin]:
        with self._lock:
            return frozenset(self._trusted_download_origins)

    def trusted_connection_endpoints(self) -> frozenset[TransportEndpoint]:
        with self._lock:
            return frozenset(self._trusted_connection_endpoints)



class SSRFGuardResolver(AbstractResolver):
    def __init__(
        self,
        *,
        policy: TransportPolicy | None = None,
        trusted_origins: set[str] | None = None,
    ) -> None:
        super().__init__()
        self.policy = policy or TransportPolicy(trusted_origins)
        self._delegate = DefaultResolver()
    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_INET,
    ) -> list[ResolveResult]:
        normalized = _canonical_trust_host(host)
        if normalized in _METADATA_HOSTS:
            raise OSError(f"unsafe host blocked by SSRF guard: {normalized}")
        trusted = self.policy.is_connection_trusted(normalized, port)
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


async def open_guarded_websocket_socket(
    url: str,
    *,
    timeout_seconds: float = 10.0,
) -> socket.socket:
    """Resolve, validate, and connect one WebSocket TCP socket without DNS rebinding."""
    origin = _canonical_origin(url)
    if origin is None or origin[0] not in {"ws", "wss"}:
        raise ValueError("WebSocket URL must use ws or wss")
    _, host, port = origin
    trusted_origins = {url} if private_hosts_enabled() else set()
    policy = TransportPolicy(trusted_origins)
    resolver = SSRFGuardResolver(policy=policy)
    last_error: OSError | None = None
    async with asyncio.timeout(timeout_seconds):
        try:
            records = await resolver.resolve(
                host,
                port,
                family=socket.AF_UNSPEC,
            )
        finally:
            await resolver.close()

        loop = asyncio.get_running_loop()
        for record in records:
            family = record["family"]
            proto = record["proto"]
            address = str(record["host"])
            connected = socket.socket(family, socket.SOCK_STREAM, proto)
            connected.setblocking(False)
            try:
                await loop.sock_connect(connected, (address, port))
            except OSError as exc:
                connected.close()
                last_error = exc
                continue
            except BaseException:
                connected.close()
                raise
            return connected

    if last_error is not None:
        raise OSError("WebSocket connection failed") from last_error
    raise OSError("WebSocket DNS resolution returned no addresses")


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
        normalized = _canonical_trust_host(host)
        trusted = self.policy.is_connection_trusted(normalized, port)
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


def new_guarded_http_session(
    *configured_urls: str,
    policy: TransportPolicy | None = None,
) -> aiohttp.ClientSession:
    if policy is None:
        trusted_origins = (
            {url for url in configured_urls if url}
            if private_hosts_enabled()
            else set()
        )
        policy = TransportPolicy(trusted_origins)
    resolver = SSRFGuardResolver(policy=policy)

    connector = SSRFGuardConnector(resolver=resolver, policy=policy)
    session = aiohttp.ClientSession(connector=connector)
    session.transport_policy = policy  # type: ignore[attr-defined]
    return session
