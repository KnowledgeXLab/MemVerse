import os
import json
import numpy as np
from datetime import datetime
from openai import OpenAI

env_base_url = os.getenv("OPENAI_API_BASE")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    **({"base_url": env_base_url} if env_base_url else {})
)

memory_files = {
    "MemoryKB/Long_Term_Memory/system/core_memory_agent.txt": "MemoryKB/Long_Term_Memory/memory_chunks/core_memory.json",
    "MemoryKB/Long_Term_Memory/system/episodic_memory_agent.txt": "MemoryKB/Long_Term_Memory/memory_chunks/episodic_memory.json",
    "MemoryKB/Long_Term_Memory/system/semantic_memory_agent.txt": "MemoryKB/Long_Term_Memory/memory_chunks/semantic_memory.json",
}


def get_embedding(text: str):
    """Call OpenAI Embedding API"""
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(resp.data[0].embedding)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# def load_existing_memory(file_path: str):
#     if not os.path.exists(file_path):
#         return []

#     memory_data = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 item = json.loads(line)
#                 if "embedding" not in item:
#                     item["embedding"] = get_embedding(item["input_text"]).tolist()
#                 memory_data.append(item)
#             except json.JSONDecodeError:
#                 continue
#     return memory_data


def process_memory(json_data: list, check_duplicate: bool = False):
    required_fields = ["id", "query", "videocaption", "audiocaption", "imagecaption"]
    for item in json_data:
        for field in required_fields:
            if field not in item:
                raise ValueError(f"Each object must have '{field}' field.")

    for prompt_file, output_file in memory_files.items():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        memory_data = []

        for entry in json_data:
            input_text_parts = [f"Query: {entry['query']}"]
            if entry.get("videocaption"):
                input_text_parts.append(f"Video: {entry['videocaption']}")
            if entry.get("audiocaption"):
                input_text_parts.append(f"Audio: {entry['audiocaption']}")
            if entry.get("imagecaption"):
                input_text_parts.append(f"Image: {entry['imagecaption']}")
            input_text = "\n".join(input_text_parts)
            new_emb = get_embedding(input_text)

            if check_duplicate and memory_data:
                # Find most similar old memory
                sims = [cosine_sim(new_emb, np.array(m["embedding"])) for m in memory_data]
                idx = int(np.argmax(sims))
                old_memory = memory_data[idx]

                # Use GPT to determine action
                instruction = (
                    "You are given two memory entries, old and new. "
                    "Classify the relationship according to the following rules:\n"
                    "- 'add': if the new memory contains any new information not present in the old memory.\n"
                    "- 'remain': if the new memory is almost identical to the old memory and contains no significant new information.\n"
                    "- 'update': if the new memory corrects errors or factual mistakes in the old memory (e.g., wrong dates, numbers, facts).\n"
                    "Respond strictly in JSON format with a single field 'action' whose value is 'add', 'update', or 'remain'.\n\n"
                    f"Old Memory:\n{old_memory['input_text']}\n\n"
                    f"New Memory:\n{input_text}\n"
                )

                resp = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "Determine redundancy of memory entries."},
                        {"role": "user", "content": instruction}
                    ],
                    temperature=0
                )

                try:
                    action_json = json.loads(resp.choices[0].message.content)
                    action = action_json.get("action", "add")
                except Exception:
                    action = "add"

                print(f"Memory id {entry['id']} action={action}")

                if action == "remain":
                    continue
                elif action == "update":
                    memory_data[idx]["input_text"] = input_text
                    memory_data[idx]["embedding"] = new_emb.tolist()
                    output_resp = client.chat.completions.create(
                        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": input_text}
                        ],
                        temperature=0
                    )
                    memory_data[idx]["output_text"] = output_resp.choices[0].message.content
                    # Save updated memory immediately
                    with open(output_file, "w", encoding="utf-8") as f:
                        for m in memory_data:
                            f.write(json.dumps(m, ensure_ascii=False) + "\n")
                    continue

            # Normal new memory
            model_to_use = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"➡️ Processing id {entry['id']} for {prompt_file} with model {model_to_use} ...")

            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input_text}
                ],
                temperature=0
            )

            output_text = response.choices[0].message.content

            memory_entry = {
                "id": entry["id"],
                "timestamp": datetime.utcnow().isoformat(),
                "input_text": input_text,
                "output_text": output_text,
                "embedding": new_emb.tolist()
            }

            memory_data.append(memory_entry)
        with open(output_file, "a", encoding="utf-8") as f:
            for m in memory_data:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        print(f"✅ {prompt_file} -> {output_file} Completed for {len(new_entries)} new entries!")


if __name__ == "__main__":
    input_file = "conversation.json"
    if not input_file.lower().endswith(".json"):
        raise ValueError("Only JSON files are supported as input.")

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON format at line {line_num}, skipping this line: {e}")
                continue

    # check_duplicate defaults to False; set True to check redundancy via GPT
    process_memory(data, check_duplicate=True)
