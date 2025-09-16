import asyncio


async def long_running_task():
    try:
        print("长时间任务开始")
        await asyncio.sleep(5)
        print("长时间任务完成")
        return "结果"
    except asyncio.CancelledError:
        print("长时间任务被取消")
        raise


async def main():
    # 使用shield
    task = asyncio.shield(long_running_task())
    await asyncio.sleep(1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("主: shield任务被取消")

    tasks = [asyncio.create_task(task(f"任务{i}"), name=f"任务{i}") for i in range(3)]

asyncio.run(main())