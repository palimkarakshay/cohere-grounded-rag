from __future__ import annotations

import argparse
import json
import sys

from .service import Service


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="grounded-rag")
    p.add_argument("--db", default=None, help="sqlite path (default from GR_DB_PATH)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest", help="ingest a directory of .md/.txt files"); s.add_argument("path")
    a = sub.add_parser("ask", help="grounded answer with citations"); a.add_argument("question"); a.add_argument("--groups", default="public")
    g = sub.add_parser("agent", help="tool-using agent run"); g.add_argument("question"); g.add_argument("--groups", default="public")
    srv = sub.add_parser("serve", help="run the HTTP API"); srv.add_argument("--host", default="127.0.0.1"); srv.add_argument("--port", type=int, default=8080)
    args = p.parse_args(argv)
    svc = Service(db_path=args.db)
    if args.cmd == "ingest":
        print(json.dumps({"chunks": svc.ingest_dir(args.path), **svc.store.count()}))
    elif args.cmd == "ask":
        print(json.dumps(svc.rag.ask(args.question, groups=args.groups.split(",")).to_dict(), indent=2))
    elif args.cmd == "agent":
        print(json.dumps(svc.agent.run(args.question, groups=args.groups.split(",")).to_dict(), indent=2))
    elif args.cmd == "serve":
        import uvicorn

        from .api import create_app
        uvicorn.run(create_app(svc), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
