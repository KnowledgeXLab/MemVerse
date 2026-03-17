# MemVerse Local Memory Skill

OpenClaw Skill for reading MemVerse memory from local MD files.

## File Structure

```
memverse-skill/
├── SKILL.md              # OpenClaw Skill definition
├── convert_json_to_md.py # JSON → MD converter script
└── README.md            # This file
```

## Installation

1. Copy to OpenClaw skills directory:
   ```bash
   cp -r memverse-skill ~/.openclaw/workspace/skills/memverse-memory
   ```

2. Run converter script (first time or updates):
   ```bash
   python3 convert_json_to_md.py
   ```

3. Restart OpenClaw Gateway

## Usage

When user asks about history, preferences, or memories, read the corresponding MD files:
- `memory/core_memory.md`     - Core memory
- `memory/semantic_memory.md` - Semantic memory  
- `memory/episodic_memory.md` - Episodic memory

## Notes

- **Sanitized**: Converter script contains no API keys
- Data source: JSON files in Docker container `memverse`
- Output format: Markdown, easy for OpenClaw Skill to read
