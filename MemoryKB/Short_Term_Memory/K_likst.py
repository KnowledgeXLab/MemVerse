import os
import json
from collections import deque

STM_FILE = "short_term_memory.jsonl"
K = int(os.getenv("STM_K", 10))

class ShortTermMemory:
    def __init__(self, file_path=STM_FILE, k=K):
        self.file_path = file_path
        self.k = k
        self.cache = deque(maxlen=k)
        self._load_cache()

    def _load_cache(self):
        if not os.path.exists(self.file_path):
            return
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    msg = json.loads(line.strip())
                    self.cache.append(msg)
                except:
                    continue

    def add_message(self, role, content):
        msg = {"role": role, "content": content}
        self.cache.append(msg)
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    def get_recent_context(self):
        return list(self.cache)

# Example usage
if __name__ == "__main__":
    stm = ShortTermMemory()

    # Add messages
    stm.add_message("user", "Hello, how are you?")
    stm.add_message("assistant", "I'm fine, thanks!")

    # Get recent K messages
    context = stm.get_recent_context()
    print(context)
