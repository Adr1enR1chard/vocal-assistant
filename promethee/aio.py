"""Run a blocking call without preventing Ctrl+C from exiting.

asyncio.to_thread uses a thread pool that asyncio.run waits for on exit: an endless input()
or microphone read then hangs the program after Ctrl+C. Here each call runs in a daemon
thread, which is abandoned on exit.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")


async def in_thread(fn: Callable[..., T], *args: Any) -> T:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()

    def settle(setter: Callable[[Any], None], value: Any) -> None:
        if not future.done():
            setter(value)

    def run() -> None:
        try:
            result = fn(*args)
        except BaseException as e:  # noqa: BLE001 — forwarded to the caller
            if not loop.is_closed():
                loop.call_soon_threadsafe(settle, future.set_exception, e)
        else:
            if not loop.is_closed():
                loop.call_soon_threadsafe(settle, future.set_result, result)

    threading.Thread(target=run, daemon=True).start()
    return await future
