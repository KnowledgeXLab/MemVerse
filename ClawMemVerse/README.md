# OpenClaw + MemVerse Memory Plugin

Use MemVerse as the long-term memory backend for [OpenClaw](https://github.com/openclaw/openclaw).

## Quick Start

```bash
# 1. Clone this repository
git clone https://github.com/KnowledgeXLab/MemVerse.git
cd MemVerse

# 2. Checkout the openmemverse branch
git checkout -b openmemverse
git pull origin openmemverse

# 3. Install and configure
mkdir -p ~/.openclaw/extensions/memverse
cp -r skill/* ~/.openclaw/extensions/memverse/
cd ~/.openclaw/extensions/memverse && npm install

# 4. Configure OpenClaw
openclaw config set plugins.enabled true
openclaw config set plugins.slots.memory memverse
openclaw config set plugins.entries.memverse.config.mode "local"
openclaw config set plugins.entries.memverse.config.containerName "memverse"
openclaw config set plugins.entries.memverse.config.autoRecall true --json
openclaw config set plugins.entries.memverse.config.autoCapture true --json

# 5. Start MemVerse container (if not running)
docker run -d --name memverse -p 8000:8000 memverse:latest

# 6. Run OpenClaw
openclaw gateway
```

## How It Works

MemVerse provides three memory layers:

| Layer | Description | Access |
|-------|-------------|--------|
| **Core Memory** | User profile, preferences, key facts | Fast retrieval |
| **Semantic Memory** | Knowledge, concepts, meanings | RAG search |
| **Episodic Memory** | Events, experiences, conversations | Time-based |

The plugin reads from local MD files exported from MemVerse Docker container.

## Manual Setup

### Prerequisites

- **Docker** with MemVerse container running on port 8000
- **OpenClaw** installed (`npm install -g openclaw`)

### Install Plugin

```bash
# Clone and checkout
git clone https://github.com/KnowledgeXLab/MemVerse.git
cd MemVerse
git checkout openmemverse

# Copy files
mkdir -p ~/.openclaw/extensions/memverse
cp -r skill/* ~/.openclaw/extensions/memverse/
cd ~/.openclaw/extensions/memverse
npm install
```

### Configure

```bash
# Enable plugin
openclaw config set plugins.enabled true
openclaw config set plugins.slots.memory memverse

# Plugin settings
openclaw config set plugins.entries.memverse.config.mode "local"
openclaw config set plugins.entries.memverse.config.containerName "memverse"
openclaw config set plugins.entries.memverse.config.autoRecall true --json
openclaw config set plugins.entries.memverse.config.autoCapture true --json
```

### Update Memory Files

To sync memories from Docker container to local MD files:

```bash
cd ~/.openclaw/extensions/memverse
python3 convert_json_to_md.py
```

This copies and converts:
- `core_memory.json` → `memory/core_memory.md`
- `semantic_memory.json` → `memory/semantic_memory.md`
- `episodic_memory.json` → `memory/episodic_memory.md`

### Start

```bash
# Ensure MemVerse is running
docker start memverse

# Start OpenClaw
openclaw gateway
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mode` | string | `"local"` | Local or remote |
| `containerName` | string | `"memverse"` | Docker container name |
| `baseUrl` | string | `"http://localhost:8000"` | MemVerse API (remote mode) |
| `apiKey` | string | `""` | API key (optional) |
| `autoRecall` | boolean | `false` | Auto-recall on startup |
| `autoCapture` | boolean | `false` | Auto-capture conversations |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Memory shows `disabled` | `openclaw config set plugins.slots.memory memverse` |
| No memories found | Run `python3 convert_json_to_md.py` to sync |
| Container not running | `docker start memverse` |
| Port 8000 in use | `docker stop $(docker ps -q --filter name=memverse)` |
| Empty memory files | Check Docker container has data: `docker exec memverse ls /app/MemoryKB/` |

## File Structure

```
MemVerse/
├── skill/
│   ├── SKILL.md              # OpenClaw skill definition
│   ├── convert_json_to_md.py # JSON to MD converter
│   └── README.md
├── memory/
│   ├── core_memory.md        # Core memory (exported)
│   ├── semantic_memory.md    # Semantic memory (exported)
│   └── episodic_memory.md    # Episodic memory (exported)
└── README.md                 # This file
```

## Docker Setup (if needed)

```bash
# Build MemVerse image
cd MemVerse
docker build -t memverse:latest .

# Run container
docker run -d --name memverse -p 8000:8000 memverse:latest

# Test
curl -s -X POST "http://localhost:8000/insert" -F "query=Hello MemVerse"
```
