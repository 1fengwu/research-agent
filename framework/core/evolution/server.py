"""零依赖 HTTP 服务（http.server），给前端联调用；生产换 FastAPI 只需换这一层。

    python -m evolve.cli serve --db data/habitfit.db --port 8000

| 方法 | 路径 | 作用 |
|---|---|---|
| GET  | /api/health | 健康检查 + 活跃版本 |
| POST | /api/runs | 新建运行 `{kind}` |
| GET  | /api/runs/{id} | 运行详情（锁定版本、三 hash、任务与结果） |
| POST | /api/runs/{id}/attempts | 上报一条结果 `{task_key,done,reason}` |
| POST | /api/runs/{id}/finish | 结束运行并返回指标 |
| POST | /api/runs/{id}/feedback | 提交反馈（幂等键 `submission_key`） |
| GET  | /api/admin/evolution | 版本链 + 摘要 + 事件 |
| POST | /api/admin/rollback | 回退 `{expected_active,target}` |
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .api import Api


def make_handler(api: Api):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # 静音，避免刷屏
            return

        def _send(self, code: int, body: dict) -> None:
            raw = json.dumps(body, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(raw)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode("utf-8"))
            except json.JSONDecodeError:
                return {}

        def do_OPTIONS(self):  # noqa: N802
            self._send(204, {})

        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0].rstrip("/")
            if path == "/api/health":
                snap = api.snapshot()
                return self._send(200, {"ok": True, "domain": api.domain.spec.name, "seq": snap.get("seq")})
            if path == "/api/admin/evolution":
                return self._send(*api.get_evolution())
            if path.startswith("/api/runs/"):
                rid = path.split("/")[3]
                return self._send(*api.get_run(rid))
            return self._send(404, {"error": "未知路径"})

        def do_POST(self):  # noqa: N802
            path = self.path.split("?")[0].rstrip("/")
            body = self._body()
            if path == "/api/runs":
                return self._send(*api.create_run(body.get("kind", "first")))
            if path == "/api/admin/rollback":
                return self._send(*api.post_rollback(body))
            parts = path.split("/")
            if len(parts) >= 5 and parts[1] == "api" and parts[2] == "runs":
                rid, action = parts[3], parts[4]
                if action == "attempts":
                    return self._send(*api.post_attempt(rid, body))
                if action == "finish":
                    return self._send(*api.finish_run(rid))
                if action == "feedback":
                    return self._send(*api.post_feedback(rid, body))
            return self._send(404, {"error": "未知路径"})

    return Handler


def serve(db: str, domain: str, subject: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    api = Api(db, domain, subject)
    srv = ThreadingHTTPServer((host, port), make_handler(api))
    print(f"evolve api  http://{host}:{port}  领域 {domain}  主体 {subject}")
    print("按 Ctrl+C 停止")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        api.close()
