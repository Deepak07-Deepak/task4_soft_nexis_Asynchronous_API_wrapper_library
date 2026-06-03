"""
github_async.py

Production-ready async GitHub API client.

Requirements:
    pip install aiohttp redis asyncio

Usage:

    import asyncio
    from github_async import GitHubAPI

    async def main():
        async with GitHubAPI(token="ghp_xxxxx") as github:

            user = await github.users.get("octocat")
            print(user)

            async for repo in github.repos.list_user_repos("octocat"):
                print(repo["name"])

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Callable,
    TypeVar,
    Awaitable,
)

import aiohttp


# ============================================================
# Exceptions
# ============================================================

class GitHubError(Exception):
    pass


class AuthenticationError(GitHubError):
    pass


class NotFoundError(GitHubError):
    pass


class RateLimitError(GitHubError):
    pass


class ServerError(GitHubError):
    pass


class ValidationError(GitHubError):
    pass


# ============================================================
# Models
# ============================================================

@dataclass(slots=True)
class GitHubUser:
    login: str
    id: int
    html_url: str

    @classmethod
    def from_dict(cls, data: dict) -> "GitHubUser":
        return cls(
            login=data["login"],
            id=data["id"],
            html_url=data["html_url"],
        )


# ============================================================
# Rate Limiter
# ============================================================

class InMemoryRateLimiter:
    """
    Simple token bucket implementation.
    """

    def __init__(
        self,
        capacity: int = 5000,
        refill_period: int = 3600,
    ):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_period = refill_period
        self.updated = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            now = time.time()

            elapsed = now - self.updated

            refill = (
                elapsed
                * self.capacity
                / self.refill_period
            )

            self.tokens = min(
                self.capacity,
                self.tokens + refill,
            )

            self.updated = now

            if self.tokens < 1:
                wait_time = (
                    self.refill_period
                    / self.capacity
                )
                await asyncio.sleep(wait_time)

            self.tokens -= 1


# ============================================================
# Retry Decorator
# ============================================================

T = TypeVar("T")


def retry(
    retries: int = 3,
    backoff: float = 1.5,
) -> Callable[
    [Callable[..., Awaitable[T]]],
    Callable[..., Awaitable[T]],
]:
    def decorator(
        func: Callable[..., Awaitable[T]]
    ) -> Callable[..., Awaitable[T]]:

        async def wrapper(*args, **kwargs) -> T:
            delay = 1.0

            for attempt in range(retries):

                try:
                    return await func(*args, **kwargs)

                except ServerError:

                    if attempt == retries - 1:
                        raise

                    await asyncio.sleep(delay)
                    delay *= backoff

            raise RuntimeError("Unexpected retry failure")

        return wrapper

    return decorator


# ============================================================
# Pagination
# ============================================================

class Paginator:
    def __init__(
        self,
        client: "GitHubAPI",
        endpoint: str,
        params: Optional[dict] = None,
    ):
        self.client = client
        self.endpoint = endpoint
        self.params = params or {}

    async def pages(self) -> AsyncGenerator[List[dict], None]:
        page = 1

        while True:

            params = {
                **self.params,
                "page": page,
            }

            data = await self.client._get(
                self.endpoint,
                params=params,
            )

            if not data:
                break

            yield data

            page += 1

    async def items(self) -> AsyncGenerator[dict, None]:
        async for page in self.pages():
            for item in page:
                yield item


# ============================================================
# Resource APIs
# ============================================================

class UsersAPI:

    def __init__(self, client: "GitHubAPI"):
        self.client = client

    async def get(self, username: str) -> GitHubUser:
        data = await self.client._get(
            f"/users/{username}"
        )
        return GitHubUser.from_dict(data)


class ReposAPI:

    def __init__(self, client: "GitHubAPI"):
        self.client = client

    async def get(
        self,
        owner: str,
        repo: str,
    ) -> dict:
        return await self.client._get(
            f"/repos/{owner}/{repo}"
        )

    async def list_user_repos(
        self,
        username: str,
        per_page: int = 100,
    ) -> AsyncGenerator[dict, None]:

        paginator = Paginator(
            self.client,
            f"/users/{username}/repos",
            {"per_page": per_page},
        )

        async for item in paginator.items():
            yield item


class IssuesAPI:

    def __init__(self, client: "GitHubAPI"):
        self.client = client

    async def list_repo_issues(
        self,
        owner: str,
        repo: str,
    ) -> AsyncGenerator[dict, None]:

        paginator = Paginator(
            self.client,
            f"/repos/{owner}/{repo}/issues",
        )

        async for issue in paginator.items():
            yield issue


# ============================================================
# Main Client
# ============================================================

class GitHubAPI:

    BASE_URL = "https://api.github.com"

    def __init__(
        self,
        token: Optional[str] = None,
        timeout: int = 30,
        rate_limiter: Optional[
            InMemoryRateLimiter
        ] = None,
    ):
        self.token = token
        self.timeout = timeout
        self.rate_limiter = (
            rate_limiter
            or InMemoryRateLimiter()
        )

        self.session: Optional[
            aiohttp.ClientSession
        ] = None

        self.users = UsersAPI(self)
        self.repos = ReposAPI(self)
        self.issues = IssuesAPI(self)

    async def __aenter__(self) -> "GitHubAPI":
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        tb,
    ):
        await self.close()

    async def open(self) -> None:

        headers = {
            "Accept":
                "application/vnd.github+json",
            "User-Agent":
                "github-async-client",
        }

        if self.token:
            headers["Authorization"] = (
                f"Bearer {self.token}"
            )

        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(
                total=self.timeout
            ),
        )

    async def close(self) -> None:

        if self.session:
            await self.session.close()

    @retry()
    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Any:

        if not self.session:
            raise RuntimeError(
                "Client not opened"
            )

        await self.rate_limiter.acquire()

        url = (
            f"{self.BASE_URL}{endpoint}"
        )

        async with self.session.request(
            method,
            url,
            **kwargs,
        ) as response:

            await self._handle_rate_limit(
                response
            )

            if response.status == 401:
                raise AuthenticationError(
                    "Invalid GitHub token"
                )

            if response.status == 404:
                raise NotFoundError(
                    f"Resource not found: "
                    f"{endpoint}"
                )

            if response.status == 422:
                raise ValidationError(
                    await response.text()
                )

            if response.status in (
                502,
                503,
                504,
            ):
                raise ServerError(
                    "GitHub unavailable"
                )

            if response.status == 429:

                retry_after = int(
                    response.headers.get(
                        "Retry-After",
                        "60",
                    )
                )

                await asyncio.sleep(
                    retry_after
                )

                raise RateLimitError(
                    "Rate limited"
                )

            response.raise_for_status()

            return await response.json()

    async def _handle_rate_limit(
        self,
        response: aiohttp.ClientResponse,
    ) -> None:

        remaining = response.headers.get(
            "X-RateLimit-Remaining"
        )

        reset = response.headers.get(
            "X-RateLimit-Reset"
        )

        if (
            remaining == "0"
            and reset
        ):
            wait = (
                int(reset)
                - int(time.time())
            )

            if wait > 0:
                await asyncio.sleep(wait)

    async def _get(
        self,
        endpoint: str,
        **kwargs,
    ) -> Any:
        return await self._request(
            "GET",
            endpoint,
            **kwargs,
        )

    async def _post(
        self,
        endpoint: str,
        **kwargs,
    ) -> Any:
        return await self._request(
            "POST",
            endpoint,
            **kwargs,
        )

    async def _patch(
        self,
        endpoint: str,
        **kwargs,
    ) -> Any:
        return await self._request(
            "PATCH",
            endpoint,
            **kwargs,
        )

    async def _delete(
        self,
        endpoint: str,
        **kwargs,
    ) -> Any:
        return await self._request(
            "DELETE",
            endpoint,
            **kwargs,
        )


# ============================================================
# Redis Rate Limiter (Optional)
# ============================================================

class RedisRateLimiter:
    """
    Production distributed rate limiter.

    Example:
        redis://localhost:6379
    """

    def __init__(
        self,
        redis_client,
        key: str = "github_rate_limit",
        limit: int = 5000,
    ):
        self.redis = redis_client
        self.key = key
        self.limit = limit

    async def acquire(self):

        current = await self.redis.incr(
            self.key
        )

        if current == 1:
            await self.redis.expire(
                self.key,
                3600,
            )

        if current > self.limit:
            ttl = await self.redis.ttl(
                self.key
            )

            await asyncio.sleep(ttl)


# ============================================================
# Package Metadata
# ============================================================

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    async def demo():

        async with GitHubAPI() as github:

            user = await github.users.get(
                "octocat"
            )

            print(user)

            async for repo in (
                github.repos.list_user_repos(
                    "octocat"
                )
            ):
                print(repo["name"])

    asyncio.run(demo())