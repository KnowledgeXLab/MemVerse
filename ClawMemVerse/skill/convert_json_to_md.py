#!/usr/bin/env python3
"""
MemVerse JSON to Markdown Converter
Converts JSON memories from Docker container to local MD files
"""

import json
import os
from datetime import datetime

MEMORY_DIR = "/Users/mini/.openclaw/workspace/memory" #Replace your working dir
CONTAINER_NAME = "memverse"
SOURCE_FILES = {
    'core_memory.json': 'Core Memory',
    'semantic_memory.json': 'Semantic Memory', 
    'episodic_memory.json': 'Episodic Memory'
}


def process_jsonl(filepath):
    """Process JSONL file (one JSON per line)"""
    memories = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    memories.append({
                        'id': item.get('id', ''),
                        'timestamp': item.get('timestamp', ''),
                        'input': item.get('input_text', ''),
                        'output': item.get('output_text', '')
                    })
                except:
                    pass
    return memories


def convert_to_markdown(memories, title):
    """Convert to Markdown format"""
    md = f"# {title}\n\n"
    md += f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    md += "---\n\n"
    
    for m in memories:
        output = m['output']
        if 'Core Memory:' in output:
            content = output.split('Core Memory:')[1].strip()
        else:
            content = output
        
        md += f"## {m['timestamp'][:10]}\n"
        md += f"**Q:** {m['input']}\n\n"
        md += f"**A:** {content}\n\n"
        md += "---\n\n"
    
    return md


def main():
    os.makedirs(MEMORY_DIR, exist_ok=True)
    
    for filename, title in SOURCE_FILES.items():
        # Copy from container
        os.system(f"docker cp {CONTAINER_NAME}:/app/MemoryKB/Long_Term_Memory/memory_chunks/{filename} /tmp/{filename}")
        
        # Process
        memories = process_jsonl(f'/tmp/{filename}')
        
        # Convert
        md = convert_to_markdown(memories, title)
        
        # Save
        md_filename = f"{MEMORY_DIR}/{filename.replace('.json', '.md')}"
        with open(md_filename, 'w') as f:
            f.write(md)
        
        print(f"✓ {title}: {len(memories)} memories -> {md_filename}")
    
    print("\nDone! You can now read the MD files directly.")


if __name__ == "__main__":
    main()
