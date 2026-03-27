import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from .tree_rag import TreeRagEngine, TreeRagNode, TreeRagSession, find_node_details, node_to_tree_dict


logger = logging.getLogger(__name__)


class TreeRagService:
    """
    Thin service wrapper around TreeRagEngine.
    Stores ONLY the last tree per (collection_name, session_id) for UI inspection.
    """

    def __init__(self, *, config: dict, rag_manager: Any, chat_service: Any):
        self._config = config
        self._rag_manager = rag_manager
        self._chat_service = chat_service
        self._sessions: dict[str, TreeRagSession] = {}

    def _key(self, collection_name: str, session_id: str) -> str:
        return f"{collection_name}::{session_id}"

    def run(
        self,
        *,
        question: str,
        session_id: str,
        collection_name: str,
        max_depth: Optional[int] = None,
    ) -> tuple[str, TreeRagNode]:
        depth = max_depth
        if depth is None:
            depth = int(self._config.get("treerag_max_depth", 2))

        max_workers = int(self._config.get("treerag_max_workers", 6))
        engine = TreeRagEngine(
            config=self._config,
            rag_manager=self._rag_manager,
            chat_service=self._chat_service,
            max_workers=max_workers,
        )

        answer, root = engine.run(
            question=question,
            session_id=session_id,
            collection_name=collection_name,
            max_depth=int(depth),
        )

        self._sessions[self._key(collection_name, session_id)] = TreeRagSession(
            updated_at=datetime.now(),
            root=root,
        )
        return answer, root

    def get_tree(self, *, collection_name: str, session_id: str) -> Optional[dict]:
        sess = self._sessions.get(self._key(collection_name, session_id))
        if not sess:
            return None
        return node_to_tree_dict(sess.root)

    def get_node(self, *, collection_name: str, session_id: str, node_id: str) -> Optional[dict]:
        sess = self._sessions.get(self._key(collection_name, session_id))
        if not sess:
            return None
        return find_node_details(sess.root, node_id)

    def cleanup_old_sessions(self):
        ttl_seconds = int(self._config.get("treerag_session_ttl_seconds", 1800))
        ttl = timedelta(seconds=max(60, ttl_seconds))
        now = datetime.now()
        to_delete = []
        for k, sess in self._sessions.items():
            if now - sess.updated_at > ttl:
                to_delete.append(k)
        for k in to_delete:
            del self._sessions[k]
        if to_delete:
            logger.info("TreeRAG cleaned up %d expired session trees", len(to_delete))

