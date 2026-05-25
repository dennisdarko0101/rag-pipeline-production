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
    from src.evaluation.metrics import RAGMetrics
    from src.generation.llm import LLMFactory

    try:
        chain = _build_rag_chain(request.provider, request.rerank)
        # LLM-as-judge for the quality metrics, using the same provider.
        judge = RAGMetrics(llm=LLMFactory.create(request.provider))
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

        # Score quality with the LLM judge. Latency above measures only the RAG
        # query; judge calls are evaluation overhead and are not counted in it.
        contexts = [s.chunk_text for s in rag_response.sources]
        faithfulness = answer_relevancy = context_precision = None
        try:
            faithfulness = judge.evaluate_faithfulness(
                pair.question, rag_response.answer, contexts
            ).score
        except Exception as e:
            logger.warning("api_eval_faithfulness_failed", error=str(e))
        try:
            answer_relevancy = judge.evaluate_answer_relevancy(
                pair.question, rag_response.answer
            ).score
        except Exception as e:
            logger.warning("api_eval_answer_relevancy_failed", error=str(e))
        try:
            context_precision = judge.evaluate_context_precision(
                pair.question, contexts, pair.ground_truth
            ).score
        except Exception as e:
            logger.warning("api_eval_context_precision_failed", error=str(e))

        results.append(
            EvalMetrics(
                question=pair.question,
                answer=rag_response.answer,
                ground_truth=pair.ground_truth,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                context_precision=context_precision,
                latency_ms=latency_ms,
                num_sources=len(rag_response.sources),
            ),
        )

    n = len(results)
    avg_latency = round(total_latency / max(n, 1), 1)

    def _avg(values: list[float | None]) -> float | None:
        present = [v for v in values if v is not None]
        return round(sum(present) / len(present), 4) if present else None

    eval_result = EvalResult(
        total_questions=n,
        avg_latency_ms=avg_latency,
        avg_faithfulness=_avg([r.faithfulness for r in results]),
        avg_answer_relevancy=_avg([r.answer_relevancy for r in results]),
        avg_context_precision=_avg([r.context_precision for r in results]),
        results=results,
    )

    eval_id = str(uuid.uuid4())
    logger.info("api_eval_complete", eval_id=eval_id, total_questions=n, avg_latency_ms=avg_latency)

    return EvalResponse(eval_id=eval_id, result=eval_result)
