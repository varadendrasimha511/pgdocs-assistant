import logging
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # the Langfuse SDK reads its keys from the environment

logger = logging.getLogger("pgdocs")


class RequestTrace:
    """One Langfuse trace per /ask request. Never raises: tracing must not break answers."""

    def __init__(self, question: str) -> None:
        self._root: Any = None
        try:
            from langfuse import get_client

            self._root = get_client().start_observation(
                name="ask", as_type="span", input={"question": question}
            )
        except Exception:
            logger.exception("tracing disabled for this request")

    def start(self, name: str, as_type: str = "span", **fields: Any) -> Any:
        if self._root is None:
            return None
        try:
            return self._root.start_observation(name=name, as_type=as_type, **fields)
        except Exception:
            logger.exception("could not start span %s", name)
            return None

    @staticmethod
    def finish(observation: Any, **fields: Any) -> None:
        if observation is None:
            return
        try:
            observation.update(**fields)
            observation.end()
        except Exception:
            logger.exception("could not end span")

    def close(self, **fields: Any) -> None:
        self.finish(self._root, **fields)


def flush_traces() -> None:
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        logger.exception("could not flush traces")