import os
from typing import Any, Optional
import asyncpg

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://asp_user:asp_password@localhost:5432/asp",
)

asyn_pool: Optional[asyncpg.Pool] = None

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


async def get_pool() -> asyncpg.Pool:
    global asyn_pool
    if asyn_pool is None:
        asyn_pool = await asyncpg.create_pool(
            DATABASE_URL, min_size=1, max_size=10, timeout=5, command_timeout=30
        )
        async with asyn_pool.acquire() as conn:
            await conn.execute(PIPE_SCHEMA)
    return asyn_pool


async def save_step(case_id: str, step: str, content: str, status: str) -> bool:
    pool = await get_pool()
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


async def load_steps(case_id: str) -> dict[str, dict[str, Any]]:
    pool = await get_pool()
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


async def delete_steps(case_id: str, steps: list[str]) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM pipeline_steps WHERE case_id = $1 AND step = ANY($2::text[])",
            case_id, steps
        )


async def list_cases(limit: int = 25) -> list[dict[str, Any]]:
    pool = await get_pool()
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

async def forget_case(case_id: str) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        n = await conn.fetchval(
            "WITH d AS (DELETE FROM pipeline_steps WHERE case_id = $1 RETURNING 1) SELECT count(*) FROM d",
            case_id
        )
    return int(n or 0)


async def delete_cases_older_than(days: int) -> int:
    pool = await get_pool()
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


async def delete_all_cases() -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("DELETE FROM pipeline_steps RETURNING case_id")
    return len({r["case_id"] for r in rows})
