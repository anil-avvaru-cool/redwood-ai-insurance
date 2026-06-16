# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
Review redwood-ai-insurance/readme.md

## Core Principles (Non-Negotiable)
* **Enterprise Architect Mode:** Prioritize architecture and decision-making before implementation. Document assumptions, trade-offs, risks, constraints, and rationale. Evaluate alternatives and recommend an approach before defining implementation details. Never begin with implementation.
* **No PII in logs, debug output, or training data**
* **No workarounds—always root-cause issues**
* **Configuration**: Use .env and example.env for configuration, never hardcoded values
* **Fail fast on config**: use `os.environ["KEY"]` (no defaults)
* **Naming standard**: underscores only (no hyphens)
* **Units**: Use USA standard units (miles, feet, pounds) in data and outputs
* **Use UV** for Python package management, running scripts, and test execution, pyproject.toml should be source of truth, do not use pip install, remove requirements.txt
* **Use Docker compose** when possible
* **Readme.md** Keep Readme.md with important info, refer granular details like Local_setup.md
Troubleshooting.md in docs folder.
* **cloud deployment** Use aws/azure deployment scripts, use environment variables for important configs
* **Do not commit to git** Because User needs to review before commit
