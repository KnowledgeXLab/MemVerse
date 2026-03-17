---
name: memverse-memory
description: "Reads MemVerse memory from local MD files. Use when user asks about past conversations, preferences, or memories."
---

# MemVerse Memory (Local MD)

Reads MemVerse memory from local MD files.

## Memory File Locations

```
memory/
├── core_memory.md     → Core Memory
├── semantic_memory.md → Semantic Memory
└── episodic_memory.md → Episodic Memory
```

## Reading Method

Use the `read` tool to read the corresponding MD files:

```
memory/core_memory.md     - User basic info, preferences
memory/semantic_memory.md - Semantic knowledge, concepts
memory/episodic_memory.md - Event memories, conversation history
```

## Trigger Conditions

Triggered when user asks about:
- "还记得..." / "do you remember..."
- "我之前说过..." / "I told you before..."
- Preferences, decisions, historical info
- Says "remember this" or "记录"

## Query Flow

1. Scan MEMORY.md index
2. Read corresponding MD file based on memory type
3. Search for relevant content in the file
4. Return results

## Example

**User asks:** "Who is the handsome guy?"

**Action:**
```bash
grep -i "guy\|handsome\|faker" memory/core_memory.md memory/semantic_memory.md
```

**Return:** Reply after finding relevant memory.

## Note

- Does not directly read JSON from Docker container
- All memories are converted to MD format and stored in workspace memory/ directory
