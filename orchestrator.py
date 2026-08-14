import asyncio
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console

from agent import Agent, MODEL, make_client
from tracer import Tracer

MAX_TURNS = 40
COLORS = {"proposer": "cyan", "skeptic": "magenta"}
console = Console()


async def get_status(session):
    result = await session.call_tool("status", {})
    return result.content[0].text


def show(role, said, actions):
    color = COLORS[role]
    if said:
        console.print(f"{role.upper()}: {said}", style=color, markup=False)
    for name, args, output in actions:
        console.print(f"   -> {name} {args}", style=color, markup=False)
        console.print(f"      {output}", style="green", markup=False)


async def investigate(claim):
    server = StdioServerParameters(command=sys.executable, args=["server.py", claim])
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            client = make_client()

            proposer = Agent("proposer", session, client, mcp_tools)
            skeptic = Agent("skeptic", session, client, mcp_tools)
            players = [proposer, skeptic]
            tracer = Tracer(claim, MODEL)

            console.rule("Investigation starting")
            console.print(claim, style="yellow", markup=False)

            for turn in range(MAX_TURNS):
                agent = players[turn % 2]
                start = time.perf_counter()
                said, actions = await agent.take_turn()
                seconds = time.perf_counter() - start
                show(agent.role, said, actions)
                status = await get_status(session)
                tracer.log(turn, agent.role, said, actions, agent.last_tokens, seconds, status)
                if status == "complete":
                    console.rule(f"Investigation complete in {turn + 1} turns")
                    final = await session.call_tool("final_report", {})
                    console.print(final.content[0].text, style="white", markup=False)
                    path = tracer.finish(status, turn + 1)
                    console.print(f"Trace saved to {path}", style="yellow", markup=False)
                    return

            console.rule("Turn limit reached - no final reports filed")
            path = tracer.finish("timeout", MAX_TURNS)
            console.print(f"Trace saved to {path}", style="yellow", markup=False)


if __name__ == "__main__":
    claim = sys.argv[1] if len(sys.argv) > 1 else "Does SMOTE improve F1 more than class-weighting on imbalanced data?"
    asyncio.run(investigate(claim))