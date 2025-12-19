import asyncio
from fastmcp.client import Client

async def main():
    async with Client("http://127.0.0.1:5250/mcp") as client:
        # 插入文本记忆
        insert_result = await client.call_tool(
            "insert_memory",
            {"query": "MemVerse is a hybrid memory system combining parametric and non-parametric memory."},
        )
        print("INSERT RESULT:", insert_result)

        # 查询记忆
        query_result = await client.call_tool(
            "query_memory",
            {"query": "What is memverse?", "mode": "hybrid", "use_pm": False},
        )
        print("QUERY RESULT:", query_result)

if __name__ == "__main__":
    asyncio.run(main())
