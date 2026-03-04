"""Evaluation endpoint: run the RAG pipeline against Q&A pairs and measure quality."""

import uuid
from time import perf_counter

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    ErrorResponse,
    EvalMetrics,
    EvalRequest,
    EvalResponse,
    EvalResult,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Evaluate"])


@router.post(
    "/evaluate",
    response_model=EvalResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Evaluation error"},
    },
    summary="Evaluate RAG quality",
    description=(
        "Run the RAG pipeline against a set of question/ground-truth pairs "
        "and return per-question and aggregate metrics."
    ),
)
async def evaluate(request: EvalRequest) -> EvalResponse:
    logger.info("api_eval_start", num_pairs=len(request.qa_pairs), provider=request.provider)

    from src.api.routes.query import _build_rag_chain

    try:
        chain = _build_rag_chain(request.provider, request.rerank)
    except Exception as e:
        logger.error("api_eval_chain_init_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to initialize RAG chain.") from e

    results: list[EvalMetrics] = []
    total_latency = 0.0

    for pair in request.qa_pairs:
        start = perf_counter()
        try:
            rag_response = chain.query(
                question=pair.question,
                k=request.k,
                rerank_top_k=5,
            )
            latency_ms = round((perf_counter() - start) * 1000, 1)
        except Exception as e:
            logger.warning("api_eval_query_failed", question=pair.question[:50], error=str(e))
            latency_ms = round((perf_counter() - start) * 1000, 1)
            results.append(
                EvalMetrics(
                    question=pair.question,
                    answer="[ERROR]",
                    ground_truth=pair.ground_truth,
                    latency_ms=latency_ms,
                    num_sources=0,
                )
            )
            total_latency += latency_ms
            continue

        total_latency += latency_ms
        results.append(
            EvalMetrics(
                question=pair.question,
                answer=rag_response.answer,
                ground_truth=pair.ground_truth,
                latency_ms=latency_ms,
                num_sources=len(rag_response.sources),
            ),
        )

    n = len(results)
    avg_latency = round(total_latency / max(n, 1), 1)

    eval_result = EvalResult(
        total_questions=n,
        avg_latency_ms=avg_latency,
        results=results,
    )

    eval_id = str(uuid.uuid4())
    logger.info("api_eval_complete", eval_id=eval_id, total_questions=n, avg_latency_ms=avg_latency)

    return EvalResponse(eval_id=eval_id, result=eval_result)
