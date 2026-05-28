from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address


def _client_ip(request) -> str:
	"""Resolve client IP behind reverse proxies (Azure ingress, etc.)."""
	xff = request.headers.get("x-forwarded-for")
	if xff:
		first = xff.split(",", 1)[0].strip()
		if first:
			return first

	xri = request.headers.get("x-real-ip")
	if xri:
		ip = xri.strip()
		if ip:
			return ip

	fallback = get_remote_address(request)

	# FastAPI TestClient reports a shared pseudo-IP "testclient" across requests,
	# which would cause unrelated tests to trip rate limits globally.
	# Keep tests isolated while preserving real per-IP limits in production.
	if fallback == "testclient":
		return f"testclient:{id(request)}"

	return fallback


limiter = Limiter(key_func=_client_ip)
