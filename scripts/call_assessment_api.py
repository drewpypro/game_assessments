#!/usr/bin/env python3
"""
Call Assessment API - Game Assessment Generator

This script:
1. Calls OpenAI API to generate a game assessment in JSON format
2. Generates a markdown report from the JSON using a Jinja2 template

Usage examples:

    # Generate assessment and markdown for a game
    python3 call_assessment_api.py --game-name "Helldivers 2"

    # With explicit Steam appid
    python3 call_assessment_api.py --game-name "Helldivers 2" --steam-appid 553850

    # Custom output paths
    python3 call_assessment_api.py --game-name "Helldivers 2" \
        --json-output examples/outputs/helldivers2_assessment_v1.json \
        --md-output docs/helldivers-2.md

    # Generate markdown only from existing JSON
    python3 call_assessment_api.py --json-only examples/outputs/helldivers2_assessment_v1.json

Requires:
    - pip install openai jinja2
    - OPENAI_API_KEY set in your environment
"""

import argparse
import json
import os
import re
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple

from jinja2 import Environment, FileSystemLoader, select_autoescape
from openai import OpenAI


class AssessmentResult(Enum):
    SUCCESS = "success"
    API_ERROR = "api_error"
    INVALID_JSON = "invalid_json"
    FILE_ERROR = "file_error"
    SCRIPT_ERROR = "script_error"


class AssessmentOutput(NamedTuple):
    result: AssessmentResult
    message: str
    json_path: str | None = None
    md_path: str | None = None
    game_name: str | None = None


ERROR_MESSAGES = {
    AssessmentResult.API_ERROR: "OpenAI API call failed. Please check your API key and try again.",
    AssessmentResult.INVALID_JSON: "Model output was not valid JSON. Please try again.",
    AssessmentResult.FILE_ERROR: "File operation failed. Please check paths and permissions.",
    AssessmentResult.SCRIPT_ERROR: "Script has errored, please review logs and contact admin.",
}


def set_github_output(name: str, value: str) -> None:
    """Set GitHub Actions output variable."""
    github_output = os.environ.get('GITHUB_OUTPUT')
    if github_output:
        with open(github_output, 'a') as f:
            f.write(f"{name}={value}\n")
    else:
        # Fallback for local testing
        print(f"::set-output name={name}::{value}")


# Resolve repo root (directory containing this script)
REPO_ROOT = Path(os.environ.get('GITHUB_WORKSPACE', Path(__file__).resolve().parent))


def slugify(text: str) -> str:
    """
    Convert text to URL-friendly slug format.
    Lower-case, replace spaces with hyphens, strip special characters.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-z0-9\-\.]", "", text)
    return text


def load_schema() -> str:
    """Load the JSON schema for game assessments."""
    schema_path = REPO_ROOT / "schema" / "game_assessment_v1.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema file: {schema_path}")
    return schema_path.read_text(encoding="utf-8")


def load_instructions() -> str:
    """
    Concatenate the instruction markdown files into a single prompt string.
    Adjust file_order if you add/remove instruction files.
    """
    instructions_dir = REPO_ROOT / "instructions"

    file_order = [
        "game_researcher_v1.md",  # top-level instructions
        "game_identity_v1.md",
        "game_anti_cheat_v1.md",
        "game_community_v1.md",
    ]

    parts = []
    for filename in file_order:
        path = instructions_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing instruction file: {path}")
        text = path.read_text(encoding="utf-8")
        parts.append(f"# {filename}\n\n{text}")

    return "\n\n\n".join(parts)


def get_slug_from_data(data: dict) -> str:
    """
    Derive a slug from the assessment JSON.
    Tries a few keys in order for future-proofing.
    """
    target = data.get("target", {}) or {}

    # Game case (current JSON structure)
    if "game_name" in target:
        return slugify(target["game_name"])

    # Web/product style options for future expansion
    for key in ("domain", "product_name", "service_name", "target_identifier"):
        if key in target and target[key]:
            return slugify(str(target[key]))

    # Fallback to generic name
    return "assessment"


def make_default_json_path(game_name: str) -> Path:
    """Generate default JSON output path based on game name."""
    slug = (
        game_name.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
    out_dir = REPO_ROOT / "examples" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{slug}_assessment_v1.json"


def generate_assessment_json(game_name: str, steam_appid: str | None = None) -> dict:
    """
    Call OpenAI API to generate a game assessment JSON.

    Args:
        game_name: Name of the game to assess
        steam_appid: Optional Steam application ID

    Returns:
        Parsed JSON assessment data as a dictionary
    """
    client = OpenAI()

    # Load instructions and schema
    instructions_text = load_instructions()
    schema_text = load_schema()

    # Build user payload
    user_payload = {"game_name": game_name}
    if steam_appid:
        user_payload["steam_appid"] = str(steam_appid)

    # Call OpenAI API
    response = client.responses.create(
        model="gpt-5.1",
        temperature=0,
        tools=[{"type": "web_search_preview"}],
        input=[
            {
                "role": "system",
                "content": (
                    instructions_text
                    + "\n\nHere is the REQUIRED JSON schema you MUST follow exactly:\n\n"
                    + schema_text
                    + "\n\nYou MUST output a JSON object matching the schema exactly. "
                    "Do not omit fields. Do NOT add fields. "
                    "Do not output markdown, explanation or prose outside JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Run the assessment using the instructions above. "
                    "Here is the input JSON for the game:\n"
                    + json.dumps(user_payload)
                ),
            },
        ],
    )

    # Extract and parse the response
    raw_text = response.output_text

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model output was not valid JSON: {e}\nRaw output:\n{raw_text}"
        )


def generate_markdown_from_json(json_data: dict, output_path: Path | None = None) -> Path:
    """
    Generate a markdown report from assessment JSON using Jinja2 template.

    Args:
        json_data: The assessment data dictionary
        output_path: Optional custom output path for the markdown file

    Returns:
        Path to the generated markdown file
    """
    template_dir = REPO_ROOT / "templates"
    template_name = "game_assessment.md.j2"
    docs_dir = REPO_ROOT / "docs"

    if not template_dir.exists():
        raise FileNotFoundError(f"Template directory does not exist: {template_dir}")

    docs_dir.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(enabled_extensions=("html", "xml")),
    )

    template = env.get_template(template_name)

    # Determine output path
    if output_path is None:
        slug = get_slug_from_data(json_data)
        output_path = docs_dir / f"{slug}.md"

    # Render and write
    rendered = template.render(**json_data)
    output_path.write_text(rendered, encoding="utf-8")

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a game assessment JSON and markdown report using OpenAI API."
    )
    parser.add_argument(
        "--game-name",
        type=str,
        help="Name of the game to assess",
    )
    parser.add_argument(
        "--steam-appid",
        type=str,
        help="Optional Steam appid for the game",
        default=None,
    )
    parser.add_argument(
        "--json-output",
        "-j",
        type=str,
        help="Path to write the JSON assessment output",
        default=None,
    )
    parser.add_argument(
        "--md-output",
        "-m",
        type=str,
        help="Path to write the markdown report output",
        default=None,
    )
    parser.add_argument(
        "--json-only",
        type=str,
        help="Path to existing JSON file - skip API call and only generate markdown",
        default=None,
    )
    parser.add_argument(
        "--skip-markdown",
        action="store_true",
        help="Skip markdown generation, only produce JSON",
    )

    args = parser.parse_args()

    try:
        # Get game_name from argument or environment variable
        if args.game_name:
            game_name = args.game_name
        else:
            game_name = os.environ.get('INPUT_GAME_NAME', '').strip()

        # Validate arguments
        if args.json_only is None and not game_name:
            print("Error: Either --game-name or --json-only must be provided", file=sys.stderr)
            set_github_output('result', AssessmentResult.SCRIPT_ERROR.value)
            set_github_output('message', ERROR_MESSAGES[AssessmentResult.SCRIPT_ERROR])
            return 1

        json_output_path = None
        md_output_path = None

        if args.json_only:
            # Load existing JSON and generate markdown only
            json_path = Path(args.json_only)
            if not json_path.exists():
                print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
                set_github_output('result', AssessmentResult.FILE_ERROR.value)
                set_github_output('message', ERROR_MESSAGES[AssessmentResult.FILE_ERROR])
                return 1

            with json_path.open("r", encoding="utf-8") as f:
                json_data = json.load(f)

            print(f"Loaded existing JSON from {json_path}")
            json_output_path = json_path
        else:
            # Generate assessment via API
            print(f"Generating assessment for: {game_name}")

            if not os.getenv("OPENAI_API_KEY"):
                print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
                set_github_output('result', AssessmentResult.API_ERROR.value)
                set_github_output('message', ERROR_MESSAGES[AssessmentResult.API_ERROR])
                return 1

            try:
                json_data = generate_assessment_json(game_name, args.steam_appid)
            except Exception as e:
                print(f"API Error: {e}", file=sys.stderr)
                set_github_output('result', AssessmentResult.API_ERROR.value)
                set_github_output('message', ERROR_MESSAGES[AssessmentResult.API_ERROR])
                return 1

            # Determine JSON output path
            if args.json_output:
                json_output_path = Path(args.json_output)
                json_output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                json_output_path = make_default_json_path(game_name)

            # Write JSON output
            output_text = json.dumps(json_data, indent=2, ensure_ascii=False)
            json_output_path.write_text(output_text, encoding="utf-8")
            print(f"Wrote JSON assessment to {json_output_path}")

        # Generate markdown unless skipped
        if not args.skip_markdown:
            md_path_arg = Path(args.md_output) if args.md_output else None
            if md_path_arg:
                md_path_arg.parent.mkdir(parents=True, exist_ok=True)

            md_output_path = generate_markdown_from_json(json_data, md_path_arg)
            print(f"Wrote markdown report to {md_output_path}")

        # Set GitHub outputs
        set_github_output('result', AssessmentResult.SUCCESS.value)
        set_github_output('message', 'Assessment completed successfully')
        set_github_output('game_name', game_name or '')
        set_github_output('json_path', str(json_output_path) if json_output_path else '')
        set_github_output('md_path', str(md_output_path) if md_output_path else '')

        print(f"Assessment Okay")
        print(f"   Game: {game_name}")
        print(f"   JSON: {json_output_path}")
        if md_output_path:
            print(f"   Markdown: {md_output_path}")

        return 0

    except Exception as e:
        print(f"Script error: {e}", file=sys.stderr)
        set_github_output('result', AssessmentResult.SCRIPT_ERROR.value)
        set_github_output('message', ERROR_MESSAGES[AssessmentResult.SCRIPT_ERROR])
        return 1


if __name__ == "__main__":
    sys.exit(main())
