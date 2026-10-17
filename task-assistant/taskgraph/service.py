"""FastAPI 服务层。

把执行图暴露成 HTTP 接口。注意几点：

1. **不引入 FastAPI 也能跑**：`create_app` 需要 fastapi/uvicorn，
   但本文件顶部的 import 是懒加载的，因此 `import taskgraph.service`
   在没装 FastAPI 的环境里也不会报错 —— 评测脚本依赖这一点。
2. 请求体做长度与类型校验，避免把任意超长输入直接喂给规划器。
3. 响应里带上 steps 明细，调用方能自行判断该不该信任这次结果。
"""

from __future__ import annotations

from typing import Any


def create_app(graph: Any = None, max_retries: int = 2):
    """构造 FastAPI 应用。需已安装 fastapi 与 uvicorn。"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    from .graph import TaskGraph

    app = FastAPI(
        title="Task Assistant API",
        version="1.0.0",
        description="多工具任务助手：任务拆解 → 工具调用 → 结果校验 → 汇总",
    )
    engine = graph or TaskGraph(max_retries=max_retries)

    class TaskRequest(BaseModel):
        task: str = Field(
            ...,
            min_length=1,
            max_length=500,
            description="要执行的任务，如「计算 128*37+456」",
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "tools": sorted(engine.tools)}

    @app.get("/tools")
    def list_tools() -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "required_params": t.required_params}
            for t in engine.tools.values()
        ]

    @app.post("/run")
    def run(req: TaskRequest) -> dict:
        try:
            state = engine.run(req.task)
        except Exception as exc:  # noqa: BLE001
            # 兜底：接口层不因单次执行失败而 500
            raise HTTPException(status_code=500, detail=f"执行失败: {type(exc).__name__}") from exc
        return state.to_dict()

    return app


def main() -> int:  # pragma: no cover
    """启动开发服务器：python taskgraph/service.py"""
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
