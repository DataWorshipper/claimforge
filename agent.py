import os
import asyncio
import httpx
from google.genai.errors import APIError
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"


ROLE_TOOLS = {
    "proposer": {
        "read_claim",
        "search_papers",
        "get_citations",
        "get_references",
        "cite",
        "send_message",
        "read_messages",
        "remember",
        "recall",
        "file_report",
        "status",
    },
    "skeptic": {
        "read_claim",
        "search_papers",
        "get_citations",
        "get_references",
        "cite",
        "run_experiment",
        "send_message",
        "read_messages",
        "remember",
        "recall",
        "file_report",
        "status",
    },
}


JSON_TO_GEMINI = {
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
}


def make_client():
    return genai.Client(api_key=API_KEY)


def build_system_prompt(role):
    shared = (
        "You and your partner are investigating a research claim together. Neither of you can "
        "reach a good answer alone. Talk with send_message and read_messages. Use remember and "
        "recall for notes. Whenever a paper actually informs your reasoning, call cite on it. "
        "Be decisive and brief. Take ONE action per turn. When you are confident in your position, "
        "call file_report with your verdict, a short summary, and the paper ids you relied on."
    )

    if role == "proposer":
        return (
            "You are the PROPOSER. "
            + shared
            + " "
            "Your job is to build the strongest honest case FOR the claim. Search literature for "
            "supporting evidence, check citations and references for corroborating or contradicting "
            "work, and share what you find with the Skeptic. You cannot run code. If you find strong "
            "evidence against the claim, report that honestly too - your goal is an accurate report, "
            "not a win."
        )

    return (
        "You are the SKEPTIC. "
        + shared
        + " "
        "Your job is to stress-test the Proposer's claim. You can search literature like they can, "
        "but you ALSO have run_experiment, which actually runs the comparison on real data. Do not "
        "just trust a paper's reported numbers - use run_experiment to check them, and use the "
        "leakage_check, seed_variance, and strengthen_baseline probes to see if the effect is real, "
        "fair, and bigger than noise. If your experiments support the claim, say so honestly - your "
        "goal is an accurate report, not to shoot it down."
    )


def to_gemini_tools(mcp_tools):
    declarations = []
    needs_agent = set()

    for tool in mcp_tools:
        schema = tool.inputSchema
        props = {}

        for name, spec in schema.get("properties", {}).items():
            if name == "agent":
                needs_agent.add(tool.name)
                continue

            props[name] = types.Schema(
                type=JSON_TO_GEMINI.get(
                    spec.get("type"),
                    "STRING",
                )
            )

        required = [
            r
            for r in schema.get("required", [])
            if r != "agent"
        ]

        parameters = (
            types.Schema(
                type="OBJECT",
                properties=props,
                required=required,
            )
            if props
            else None
        )

        declarations.append(
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "",
                parameters=parameters,
            )
        )

    return [
        types.Tool(
            function_declarations=declarations
        )
    ], needs_agent


def tool_result_text(result):
    return "\n".join(
        block.text
        for block in result.content
        if hasattr(block, "text")
    )


async def generate_with_retry(client, **kwargs):
    for attempt in range(6):
        try:
            return await client.aio.models.generate_content(
                **kwargs
            )

        except APIError as error:
            wait = 20 if error.code == 429 else 10

            if attempt < 5:
                print(f"API error {error.code} ({error.status}), retrying in {wait}s...")
                await asyncio.sleep(wait)
                continue

            raise

        except httpx.TransportError as error:
            if attempt < 5:
                print(f"network hiccup ({error}), retrying in 5s...")
                await asyncio.sleep(5)
                continue

            raise


class Agent:
    def __init__(
        self,
        role,
        session,
        client,
        mcp_tools,
    ):
        self.role = role
        self.session = session
        self.client = client

        selected = [
            t
            for t in mcp_tools
            if t.name in ROLE_TOOLS[role]
        ]

        self.tools, self.needs_agent = to_gemini_tools(
            selected
        )

        self.system = build_system_prompt(role)

        self.contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text="Begin investigating the claim."
                    )
                ],
            )
        ]

        self.last_tokens = 0

    def nudge(self, text):
        self.contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=text)],
            )
        )

    async def call_tool(self, name, arguments):
        args = dict(arguments)

        if name in self.needs_agent:
            args["agent"] = self.role

        result = await self.session.call_tool(
            name,
            args,
        )

        return tool_result_text(result)

    async def take_turn(self):
        response = await generate_with_retry(
            self.client,
            model=MODEL,
            contents=self.contents,
            config=types.GenerateContentConfig(
                system_instruction=self.system,
                tools=self.tools,
            ),
        )

        self.last_tokens = (
            response.usage_metadata.total_token_count
        )

        reply = response.candidates[0].content
        self.contents.append(reply)

        said = ""
        actions = []
        responses = []

        for part in reply.parts or []:
            if part.text:
                said += part.text

            if part.function_call:
                call = part.function_call
                args = dict(call.args)

                output = await self.call_tool(
                    call.name,
                    args,
                )

                responses.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"result": output},
                    )
                )

                actions.append(
                    (
                        call.name,
                        args,
                        output,
                    )
                )

        if responses:
            self.contents.append(
                types.Content(
                    role="user",
                    parts=responses,
                )
            )
        else:
            self.contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text="Take an action using one of your tools."
                        )
                    ],
                )
            )

        return said.strip(), actions