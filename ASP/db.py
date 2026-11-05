import asyncio
import os
from typing import Any, Optional
import asyncpg

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://asp_user:asp_password@localhost:5432/asp",
)

PIPE_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_steps (
    case_id     TEXT NOT NULL,
    step        TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('partial', 'complete')),
    content     TEXT NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (case_id, step)
);
"""


class CheckpointDB:
    def __init__(
        self,
        url: str = DATABASE_URL,
        min_size: int = 1,
        max_size: int = 10,
        timeout: float = 5,
        command_timeout: float = 30,
    ):
        self.url = url
        self.min_size = min_size
        self.max_size = max_size
        self.timeout = timeout
        self.command_timeout = command_timeout
        self._pool: Optional[asyncpg.Pool] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock: Optional[asyncio.Lock] = None
        self._lock_loop: Optional[asyncio.AbstractEventLoop] = None

    def lock_for(self, loop: asyncio.AbstractEventLoop) -> asyncio.Lock:
        if self._lock is None or self._lock_loop is not loop:
            self._lock, self._lock_loop = asyncio.Lock(), loop
        return self._lock

    async def pool(self) -> asyncpg.Pool:
        loop = asyncio.get_running_loop()
        if self._pool is not None and self._loop is loop:
            return self._pool
        async with self.lock_for(loop):
            if self._pool is None or self._loop is not loop:
                self.discard_pool()
                pool = await asyncpg.create_pool(
                    self.url,
                    min_size=self.min_size,
                    max_size=self.max_size,
                    timeout=self.timeout,
                    command_timeout=self.command_timeout,
                )
                async with pool.acquire() as conn:
                    await conn.execute(PIPE_SCHEMA)
                self._pool, self._loop = pool, loop
            return self._pool

    def discard_pool(self) -> None:
        stale, self._pool, self._loop = self._pool, None, None
        if stale is not None:
            try:
                stale.terminate()
            except Exception:
                pass

    async def close(self) -> None:
        pool, self._pool, self._loop = self._pool, None, None
        if pool is not None:
            await pool.close()

    async def save_step(self, case_id: str, step: str, content: str, status: str) -> bool:
        pool = await self.pool()
        async with pool.acquire() as conn:
            row = await conn.fetchval(
                """
                INSERT INTO pipeline_steps (case_id, step, status, content, updated_at)
                VALUES ($1, $2, $3, $4, now())
                ON CONFLICT (case_id, step) DO UPDATE
                   SET status = EXCLUDED.status,
                       content = EXCLUDED.content,
                       updated_at = now()
                 WHERE NOT (pipeline_steps.status = 'complete' AND EXCLUDED.status = 'partial')
                RETURNING 1
                """,
                case_id, step, status, content
            )
        return row is not None

    async def load_steps(self, case_id: str) -> dict[str, dict[str, Any]]:
        pool = await self.pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT step, status, content, updated_at FROM pipeline_steps WHERE case_id = $1",
                case_id
            )
        return {
            r["step"]: {
                "status": r["status"],
                "content": r["content"],
                "updated_at": r["updated_at"].isoformat()
            }
            for r in rows
        }

    async def delete_steps(self, case_id: str, steps: list[str]) -> None:
        pool = await self.pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM pipeline_steps WHERE case_id = $1 AND step = ANY($2::text[])",
                case_id, steps
            )

    async def list_cases(self, limit: int = 25) -> list[dict[str, Any]]:
        pool = await self.pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT case_id,
                       max(updated_at) AS updated_at,
                       string_agg(step || ':' || status, ',' ORDER BY step) AS steps,
                       max(content) FILTER (WHERE step = 'meta') AS meta
                  FROM pipeline_steps
                 GROUP BY case_id
                 ORDER BY max(updated_at) DESC
                 LIMIT $1
                """,
                limit
            )
        return [
            {
                "case_id": r["case_id"],
                "updated_at": r["updated_at"].isoformat(),
                "steps": r["steps"],
                "meta": r["meta"]
            }
            for r in rows
        ]

    async def forget_case(self, case_id: str) -> int:
        pool = await self.pool()
        async with pool.acquire() as conn:
            n = await conn.fetchval(
                "WITH d AS (DELETE FROM pipeline_steps WHERE case_id = $1 RETURNING 1) SELECT count(*) FROM d",
                case_id
            )
        return int(n or 0)

    async def delete_cases_older_than(self, days: int) -> int:
        pool = await self.pool()
        async with pool.acquire() as conn:
            ids = [
                r["case_id"]
                for r in await conn.fetch(
                    """
                    DELETE FROM pipeline_steps WHERE case_id IN (
                        SELECT case_id FROM pipeline_steps
                         GROUP BY case_id
                        HAVING max(updated_at) < now() - make_interval(days => $1)
                    ) RETURNING case_id
                    """,
                    int(days)
                )
            ]
        return len(set(ids))

    async def delete_all_cases(self) -> int:
        pool = await self.pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("DELETE FROM pipeline_steps RETURNING case_id")
        return len({r["case_id"] for r in rows})


checkpoint_db = CheckpointDB()
