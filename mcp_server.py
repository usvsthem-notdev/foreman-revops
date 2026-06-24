"""
Foreman MCP server — exposes LLM spend analytics as tools for Cursor / Claude.

Run via stdio (Cursor MCP config):
  {
    "mcpServers": {
      "foreman": {
        "command": "/path/to/foreman-revops/.venv/bin/python",
        "args": ["/path/to/foreman-revops/mcp_server.py"],
        "env": { "FOREMAN_DB_PATH": "/path/to/foreman.db" }
      }
    }
  }
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.analytics.burn_map import (
    burn_by_class,
    burn_by_model,
    burn_by_provider,
    burn_rate_projection,
    cumulative_burn,
    daily_burn,
    key_metrics,
    load_dataframe,
)
from src.db import fetch_budgets

server = Server("foreman-revops")


def _df(days: int | None = None):
    df = load_dataframe()
    if days and not df.empty:
        cutoff = datetime.utcnow() - timedelta(days=days)
        df = df[df["timestamp"] >= cutoff]
    return df


def _json(obj) -> str:
    return json.dumps(obj, default=str, indent=2)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_key_metrics",
            description=(
                "Get headline LLM spend metrics: total cost, frontier vs absorbed split, "
                "token counts, cost per 1K tokens, and entry count. "
                "Use this for a quick spend snapshot."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Lookback window in days (omit for all-time)",
                    }
                },
            },
        ),
        Tool(
            name="get_burn_by_provider",
            description="Get total spend broken down by provider (Anthropic, OpenAI, Cursor, Gemini, etc.).",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Lookback window in days"}
                },
            },
        ),
        Tool(
            name="get_burn_by_model",
            description="Get top models ranked by total spend.",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Lookback window in days"},
                    "top_n": {"type": "integer", "description": "Number of models to return (default 10)"},
                },
            },
        ),
        Tool(
            name="get_burn_by_class",
            description=(
                "Get spend by workload class (extract, rag, reason, agents, coding). "
                "Each class shows frontier spend and absorbed (local model) spend separately."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Lookback window in days"}
                },
            },
        ),
        Tool(
            name="get_daily_burn",
            description="Get day-by-day spend for the past N days, split by frontier and absorbed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Number of days to show (default 14)"}
                },
            },
        ),
        Tool(
            name="get_projection",
            description=(
                "Get a spend projection: daily average burn rate and projected total "
                "over the next N days based on recent trend."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "days_ahead": {
                        "type": "integer",
                        "description": "Days to project forward (default 30)",
                    },
                    "lookback_days": {
                        "type": "integer",
                        "description": "Days of history to base the projection on (default 30)",
                    },
                },
            },
        ),
        Tool(
            name="get_budget_status",
            description="Get configured budgets and how much of each has been spent.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_top_spenders",
            description=(
                "Get the top teams or features by spend. Useful for identifying "
                "which part of the org is driving cost."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "enum": ["team", "feature"],
                        "description": "Dimension to group by",
                    },
                    "days": {"type": "integer", "description": "Lookback window in days"},
                    "top_n": {"type": "integer", "description": "Number of results (default 10)"},
                },
                "required": ["group_by"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    days = arguments.get("days")

    if name == "get_key_metrics":
        df = _df(days)
        m = key_metrics(df)
        label = f"last {days} days" if days else "all time"
        result = {
            "period": label,
            "total_cost_usd": round(m["total_cost_usd"], 4),
            "frontier_cost_usd": round(m["frontier_cost_usd"], 4),
            "absorbed_cost_usd": round(m["absorbed_cost_usd"], 4),
            "local_pct": round(m["local_pct"], 1),
            "entry_count": m["entry_count"],
            "total_input_tokens": m["total_input_tokens"],
            "total_output_tokens": m["total_output_tokens"],
            "cost_per_1k_tokens": round(m["cost_per_1k_tokens"], 6),
        }
        return [TextContent(type="text", text=_json(result))]

    if name == "get_burn_by_provider":
        df = _df(days)
        rows = burn_by_provider(df)
        result = rows[["provider", "total_usd"]].round(4).to_dict(orient="records")
        return [TextContent(type="text", text=_json(result))]

    if name == "get_burn_by_model":
        df = _df(days)
        top_n = arguments.get("top_n", 10)
        rows = burn_by_model(df, top_n=top_n)
        result = rows[["model", "total_usd"]].round(4).to_dict(orient="records")
        return [TextContent(type="text", text=_json(result))]

    if name == "get_burn_by_class":
        df = _df(days)
        rows = burn_by_class(df)
        cols = [c for c in ["workload_class", "frontier_usd", "absorbed_usd", "total_usd"] if c in rows.columns]
        result = rows[cols].round(4).to_dict(orient="records")
        return [TextContent(type="text", text=_json(result))]

    if name == "get_daily_burn":
        n = days or 14
        df = _df(n)
        rows = daily_burn(df)
        cols = [c for c in ["date", "frontier_usd", "absorbed_usd", "total_usd"] if c in rows.columns]
        result = rows[cols].round(4).to_dict(orient="records")
        return [TextContent(type="text", text=_json(result))]

    if name == "get_projection":
        lookback = arguments.get("lookback_days", 30)
        days_ahead = arguments.get("days_ahead", 30)
        df = _df(lookback)
        proj = burn_rate_projection(df, days_ahead=days_ahead)
        result = {
            "daily_avg_usd": round(proj["daily_avg"], 4),
            "days_of_data": proj["days_of_data"],
            "days_ahead": days_ahead,
            "projected_total_usd": round(proj["projected_total"], 4),
        }
        return [TextContent(type="text", text=_json(result))]

    if name == "get_budget_status":
        from src.analytics.burn_map import budget_status
        df = _df()
        budgets = fetch_budgets()
        statuses = budget_status(df, [b.model_dump() for b in budgets])
        for s in statuses:
            for k in ("spent_usd", "remaining_usd", "pct_used"):
                if k in s:
                    s[k] = round(float(s[k]), 4)
        return [TextContent(type="text", text=_json(statuses))]

    if name == "get_top_spenders":
        group_by = arguments.get("group_by", "team")
        top_n = arguments.get("top_n", 10)
        df = _df(days)
        if df.empty or group_by not in df.columns:
            return [TextContent(type="text", text="[]")]
        result = (
            df.groupby(group_by, dropna=False)["cost_usd"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .round(4)
            .reset_index()
            .rename(columns={"cost_usd": "total_usd"})
            .to_dict(orient="records")
        )
        return [TextContent(type="text", text=_json(result))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
