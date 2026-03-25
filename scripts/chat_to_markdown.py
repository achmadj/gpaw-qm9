#!/usr/bin/env python3
"""
Convert chat.json from GitHub Copilot to markdown format.
Preserves user messages, AI thinking, tool invocations, and command outputs.
"""

import json
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = str(PROJECT_ROOT / "gpaw_analysis_outputs" / "chat" / "chat.json")
OUTPUT_FILE = str(PROJECT_ROOT / "gpaw_analysis_outputs" / "chat" / "chat.md")


def format_timestamp(ts_ms):
    """Convert milliseconds timestamp to readable format."""
    if ts_ms:
        try:
            dt = datetime.fromtimestamp(ts_ms / 1000)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return str(ts_ms)
    return "Unknown"


def clean_text(text):
    """Clean text for markdown output."""
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_parts(parts):
    """Extract text from message parts."""
    texts = []
    for part in parts:
        if isinstance(part, dict):
            if part.get("kind") == "text":
                texts.append(part.get("text", ""))
            elif part.get("text"):
                texts.append(part.get("text", ""))
    return "".join(texts)


def parse_response_item(item, indent=0):
    """Parse a single response item and return formatted markdown."""
    if not isinstance(item, dict):
        return str(item)

    kind = item.get("kind", "")
    prefix = "  " * indent

    if kind == "thinking":
        value = item.get("value", "")
        if value:
            return f"**Thinking:**\n{value}\n"
        return ""

    elif kind == "text":
        text = item.get("text", "")
        return clean_text(text)

    elif kind == "toolInvocationSerialized":
        # Tool invocation - extract command and output
        inv_msg = item.get("invocationMessage", "")
        result_msg = item.get("resultMessage", "")

        output = f"{prefix}**Tool Invocation:**\n"
        if inv_msg:
            output += f"{prefix}Command: `{inv_msg}`\n"
        if result_msg:
            output += (
                f"{prefix}Output:\n{prefix}```\n{prefix}{result_msg}\n{prefix}```\n"
            )
        return output

    elif kind == "toolUseResult":
        # Tool result
        content = item.get("content", [])
        output = f"{prefix}**Tool Result:**\n"
        for c in content:
            if isinstance(c, dict):
                if c.get("kind") == "text":
                    output += f"{prefix}{clean_text(c.get('text', ''))}\n"
        return output

    elif kind == "image":
        # Image reference
        uri = item.get("uri", "")
        alt = item.get("altText", "image")
        return f"{prefix}![{alt}]({uri})\n"

    return ""


def parse_request(request, turn_num):
    """Parse a single request/turn and return formatted markdown."""
    md = []

    # Header
    timestamp = request.get("timestamp", "")
    md.append(f"## Turn {turn_num} ({format_timestamp(timestamp)})")
    md.append("")

    # User message
    message = request.get("message", {})
    user_text = ""

    if isinstance(message, dict):
        user_text = message.get("text", "")
        if not user_text:
            # Try parts
            parts = message.get("parts", [])
            user_text = extract_text_from_parts(parts)

    if user_text:
        md.append("### User")
        md.append("")
        md.append(clean_text(user_text))
        md.append("")

    # Response items
    response = request.get("response", [])
    if response:
        md.append("---")
        md.append("")
        md.append("### AI Response")
        md.append("")

        for resp_item in response:
            parsed = parse_response_item(resp_item)
            if parsed:
                md.append(parsed)

    # Result (execution details)
    result = request.get("result", {})
    if result:
        # Result can be string or dict
        if isinstance(result, str):
            md.append(f"**Result:** {result}")
        elif isinstance(result, dict):
            details = result.get("details", {})
            if details:
                md.append("")
                md.append("**Execution Details:**")
                if isinstance(details, dict):
                    md.append(f"- Time: {details.get('time', 'N/A')}")

    return "\n".join(md)


def main():
    print(f"Loading {INPUT_FILE}...")

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    responder = data.get("responderUsername", "Unknown")
    requests = data.get("requests", [])

    print(f"Found {len(requests)} conversation turns")

    # Build markdown
    md_lines = []
    md_lines.append(f"# Conversation with {responder}")
    md_lines.append("")
    md_lines.append(f"*Extracted from chat.json*")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    for i, req in enumerate(requests, 1):
        parsed = parse_request(req, i)
        md_lines.append(parsed)
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    # Write output
    output = "\n".join(md_lines)
    with open(OUTPUT_FILE, "w") as f:
        f.write(output)

    print(f"Done! Written to {OUTPUT_FILE}")
    print(f"Output size: {len(output)} characters")


if __name__ == "__main__":
    main()
