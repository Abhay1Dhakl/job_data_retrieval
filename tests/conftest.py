import os
import sys
import types


def _ensure_pinecone_stub() -> None:
    try:
        import pinecone as _pinecone  # type: ignore

        if hasattr(_pinecone, "Pinecone") and hasattr(_pinecone, "ServerlessSpec"):
            return
    except Exception:
        pass

    class _StubIndex:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Pinecone client not available in test environment.")

    class Pinecone:
        def __init__(self, *args, **kwargs):
            pass

        def list_indexes(self):
            return {"indexes": []}

        def create_index(self, *args, **kwargs):
            return None

        def Index(self, *args, **kwargs):
            return _StubIndex()

    class ServerlessSpec:
        def __init__(self, cloud: str, region: str):
            self.cloud = cloud
            self.region = region

    stub = types.ModuleType("pinecone")
    stub.Pinecone = Pinecone
    stub.ServerlessSpec = ServerlessSpec
    sys.modules["pinecone"] = stub

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

_ensure_pinecone_stub()
