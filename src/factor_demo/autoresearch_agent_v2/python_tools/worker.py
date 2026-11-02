"""在 Jupyter kernel 中安装类代理，真实工具对象始终保留在宿主。"""

from collections.abc import Callable
from typing import Any


def install(classes: dict[str, list[str]], namespace: dict[str, Any]) -> None:
    """只向 Notebook 注入已加载类的同步调用代理。

    Args:
        classes: 宿主提供的类名及公开方法名，不包含方法实现或 Schema。
        namespace: 当前 Notebook 的持久变量空间。
    """
    import builtins
    import json
    import threading

    # 所有线程共用一条 stdin 请求响应通道，不向 Notebook 注入宿主对象。
    call_lock = threading.Lock()
    request_input = builtins.input

    def method_proxy(class_name: str, method_name: str) -> Callable[..., Any]:
        """为一个已加载方法创建不包含真实签名的代理。"""

        def call(*args: Any, **kwargs: Any) -> Any:
            """提交有界 JSON 请求，交由宿主校验并调用工具。"""
            payload = json.dumps(
                {
                    "class_name": class_name,
                    "method_name": method_name,
                    "args": list(args),
                    "kwargs": kwargs,
                },
                ensure_ascii=False,
                allow_nan=False,
            )
            if len(payload.encode("utf-8")) > 1024 * 1024:
                raise ValueError("单次工具请求超过 1 MiB，请缩小参数或使用资源引用。")
            with call_lock:
                response = json.loads(request_input("\x1ePYTHON_TOOL:" + payload))
            if "error" in response:
                error = response["error"]
                raise RuntimeError(f"{error['type']}: {error['message']}")
            return response["result"]

        return call

    for class_name, methods in classes.items():
        namespace[class_name] = type(
            class_name,
            (),
            {method: staticmethod(method_proxy(class_name, method)) for method in methods},
        )
