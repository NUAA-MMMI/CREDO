"""Canonical CF-PSR method implementation shared by private platform runtimes.

Coverage-Frontier PSR uses evaluator-semantics-aligned target aggregation.

The module is deliberately independent from the historical PSR implementation.
It runs on the complete driver-side rollout batch, never changes the scalar
answer reward, and emits bounded positive local credits plus a normalized
within-prompt stage-routing coefficient for the actor.

The input schema's ``ground_truth.target`` entries are acceptable answers (the official
EM accepts any one of them), not a set of supporting facts that all have to be
recovered.  Coverage is therefore computed independently for every acceptable
target and reduced with ``max``.  This keeps the dense proxy aligned with the
official evaluator and avoids rewarding retrieval of every alias.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

CF_STAGES = ("query", "evidence", "reasoning")
CF_VALIDITY_MODES = ("strict_exclusion", "stage_local_valid")
_EPS = 1.0e-8
_TRACKED_TAGS = ("search", "information", "filter", "rank", "think", "answer")
_DOC_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]\s*|Doc\s+\d+\s*(?:\(Title:\s*)?)", re.IGNORECASE)


@dataclass
class CFSpanParse:
    """CF-owned, tokenizer-aligned decision-span parse.

    The parser lives with the CF estimator so the experiment does not import
    other experimental parsing code.  Environment-injected information is tracked
    separately and can never be routed local policy weight.
    """

    query_mask: torch.Tensor
    evidence_mask: torch.Tensor
    reasoning_mask: torch.Tensor
    other_actor_mask: torch.Tensor
    information_mask: torch.Tensor
    parse_reliable: bool
    failure_reason: str
    response_text: str
    alignment_method: str
    search_transition_complete: bool = False
    search_transition_failure_reason: str = ""
    stage_parse_reliable: dict[str, bool] | None = None
    stage_failure_reason: dict[str, str] | None = None


def _empty_span_parse(
    length: int,
    actor_loss_mask: torch.Tensor,
    reason: str,
    text: str = "",
) -> CFSpanParse:
    zeros = torch.zeros(length, dtype=torch.float32)
    actor = actor_loss_mask.detach().cpu().to(torch.float32)
    return CFSpanParse(
        query_mask=zeros.clone(),
        evidence_mask=zeros.clone(),
        reasoning_mask=zeros.clone(),
        other_actor_mask=actor.clone(),
        information_mask=zeros.clone(),
        parse_reliable=False,
        failure_reason=reason,
        response_text=text,
        alignment_method="none",
        search_transition_complete=False,
        search_transition_failure_reason=reason,
        stage_parse_reliable={stage: False for stage in CF_STAGES},
        stage_failure_reason={stage: reason for stage in CF_STAGES},
    )


def _tokenizer_call(tokenizer, text: str, **kwargs) -> dict[str, Any]:
    encoded = tokenizer(text, **kwargs)
    if hasattr(encoded, "data"):
        encoded = encoded.data
    return dict(encoded)


def _align_response_tokens(
    tokenizer,
    token_ids: Sequence[int],
) -> tuple[str, list[tuple[int, int]], str, str]:
    """Align rollout IDs to characters without accepting silent re-encoding drift."""

    ids = [int(value) for value in token_ids]
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    try:
        encoded = _tokenizer_call(
            tokenizer,
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded_ids = [int(value) for value in encoded.get("input_ids", [])]
        offsets = [tuple(map(int, pair)) for pair in encoded.get("offset_mapping", [])]
        if encoded_ids == ids and len(offsets) == len(ids):
            return text, offsets, "fast_exact", ""
    except (TypeError, ValueError, NotImplementedError):
        pass

    offsets: list[tuple[int, int]] = []
    previous = ""
    for end_index in range(1, len(ids) + 1):
        prefix = tokenizer.decode(
            ids[:end_index],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if len(prefix) < len(previous) or not prefix.startswith(previous) or not text.startswith(prefix):
            return text, [], "none", "token_alignment_failed"
        offsets.append((len(previous), len(prefix)))
        previous = prefix
    if previous != text or len(offsets) != len(ids):
        return text, [], "none", "token_alignment_failed"
    return text, offsets, "prefix_exact", ""


def _tag_blocks(text: str, tag: str) -> list[re.Match[str]]:
    return list(re.finditer(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE))


def _tag_validation_errors(text: str) -> dict[str, str]:
    errors: dict[str, str] = {}
    for tag in _TRACKED_TAGS:
        opening = len(re.findall(rf"<{tag}>", text, flags=re.IGNORECASE))
        closing = len(re.findall(rf"</{tag}>", text, flags=re.IGNORECASE))
        complete = len(_tag_blocks(text, tag))
        if opening != closing or complete != opening:
            errors[tag] = f"malformed_{tag}_tag"
    if not _tag_blocks(text, "answer"):
        errors.setdefault("answer", "missing_answer")
    return errors


def _search_transition_status(
    search_blocks: Sequence[re.Match[str]],
    information_blocks: Sequence[re.Match[str]],
) -> tuple[bool, str]:
    if not search_blocks:
        return False, "missing_search_action"
    for index, search in enumerate(search_blocks):
        boundary = search_blocks[index + 1].start() if index + 1 < len(search_blocks) else math.inf
        if not any(search.end() <= information.start() < boundary for information in information_blocks):
            return False, "search_without_information_state"
    return True, ""


def _mask_char_spans(
    offsets: Sequence[tuple[int, int]],
    spans: Iterable[tuple[int, int]],
    length: int,
    *,
    require_contained: bool = True,
) -> torch.Tensor:
    mask = torch.zeros(length, dtype=torch.float32)
    spans = list(spans)
    for token_index, (start, end) in enumerate(offsets):
        if token_index >= length or end <= start:
            continue
        if require_contained:
            selected = any(span_start <= start and end <= span_end for span_start, span_end in spans)
        else:
            selected = any(start < span_end and span_start < end for span_start, span_end in spans)
        if selected:
            mask[token_index] = 1.0
    return mask


def parse_cf_psr_spans(
    tokenizer,
    response_ids: Sequence[int] | torch.Tensor,
    actor_loss_mask: torch.Tensor,
    *,
    strict: bool = True,
    validity_mode: str = "strict_exclusion",
) -> CFSpanParse:
    """Parse disjoint Query/Evidence/Reasoning spans for CF-PSR only."""

    if validity_mode not in CF_VALIDITY_MODES:
        raise ValueError(f"Unsupported CF validity_mode={validity_mode!r}")
    stage_local = validity_mode == "stage_local_valid"
    actor = actor_loss_mask.detach().cpu().to(torch.float32)
    length = int(actor.numel())
    ids = response_ids.detach().cpu().tolist() if isinstance(response_ids, torch.Tensor) else list(response_ids)
    ids = ids[:length]
    text, offsets, alignment_method, alignment_error = _align_response_tokens(tokenizer, ids)
    if alignment_error:
        return _empty_span_parse(length, actor, alignment_error, text)

    tag_errors = _tag_validation_errors(text)
    tag_error = next(iter(tag_errors.values()), "")
    if strict and tag_error and not stage_local:
        return _empty_span_parse(length, actor, tag_error, text)

    search_blocks = _tag_blocks(text, "search")
    information_blocks = _tag_blocks(text, "information")
    filter_blocks = _tag_blocks(text, "filter")
    rank_blocks = _tag_blocks(text, "rank")
    think_blocks = _tag_blocks(text, "think")
    answer_blocks = _tag_blocks(text, "answer")
    transition_complete, transition_failure = _search_transition_status(search_blocks, information_blocks)

    query_spans = [match.span(1) for match in search_blocks if match.group(1)]
    evidence_spans = [
        match.span(1) for match in (*filter_blocks, *rank_blocks) if match.group(1)
    ]
    information_spans = [match.span(0) for match in information_blocks]
    if information_blocks:
        reasoning_boundary = information_blocks[-1].end()
    elif search_blocks:
        reasoning_boundary = search_blocks[-1].end()
    else:
        reasoning_boundary = 0
    reasoning_spans = [
        match.span(1)
        for match in think_blocks
        if match.start() >= reasoning_boundary and match.group(1)
    ]
    reasoning_spans.extend(match.span(1) for match in answer_blocks if match.group(1))

    query_mask = _mask_char_spans(offsets, query_spans, length) * actor
    evidence_mask = _mask_char_spans(offsets, evidence_spans, length) * actor
    reasoning_mask = _mask_char_spans(offsets, reasoning_spans, length) * actor
    information_mask = _mask_char_spans(offsets, information_spans, length, require_contained=False)
    overlap = query_mask * evidence_mask + query_mask * reasoning_mask + evidence_mask * reasoning_mask
    stage_union = torch.clamp(query_mask + evidence_mask + reasoning_mask, 0.0, 1.0)
    if overlap.sum().item() != 0:
        return _empty_span_parse(length, actor, "stage_mask_overlap", text)
    if (stage_union * (1.0 - actor)).sum().item() != 0:
        return _empty_span_parse(length, actor, "stage_mask_outside_actor_mask", text)
    if (stage_union * information_mask).sum().item() != 0:
        return _empty_span_parse(length, actor, "information_stage_overlap", text)

    stage_reliable = {
        "query": "search" not in tag_errors and "information" not in tag_errors,
        "evidence": "filter" not in tag_errors and "rank" not in tag_errors,
        "reasoning": "think" not in tag_errors and "answer" not in tag_errors and bool(answer_blocks),
    }
    stage_failure = {
        "query": tag_errors.get("search", tag_errors.get("information", "")),
        "evidence": tag_errors.get("filter", tag_errors.get("rank", "")),
        "reasoning": tag_errors.get("think", tag_errors.get("answer", "")),
    }
    reliable = not tag_error
    if not reliable and not stage_local:
        return _empty_span_parse(length, actor, tag_error or "span_parse_failed", text)
    if stage_local:
        if not stage_reliable["query"]:
            query_mask.zero_()
        if not stage_reliable["evidence"]:
            evidence_mask.zero_()
        if not stage_reliable["reasoning"]:
            reasoning_mask.zero_()
        stage_union = torch.clamp(query_mask + evidence_mask + reasoning_mask, 0.0, 1.0)

    return CFSpanParse(
        query_mask=query_mask,
        evidence_mask=evidence_mask,
        reasoning_mask=reasoning_mask,
        other_actor_mask=torch.clamp(actor - stage_union, min=0.0, max=1.0),
        information_mask=information_mask,
        parse_reliable=reliable,
        failure_reason=tag_error,
        response_text=text,
        alignment_method=alignment_method,
        search_transition_complete=transition_complete,
        search_transition_failure_reason=transition_failure,
        stage_parse_reliable=stage_reliable,
        stage_failure_reason=stage_failure,
    )


def _uid_key(uid: Any) -> tuple[str, Any]:
    if hasattr(uid, "item"):
        try:
            uid = uid.item()
        except ValueError:
            pass
    try:
        hash(uid)
        value = uid
    except TypeError:
        value = repr(uid)
    return type(uid).__qualname__, value


def _tag_lines(text: str, tag: str, *, last_only: bool = False) -> list[str]:
    blocks = re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    if last_only and blocks:
        blocks = blocks[-1:]
    return [line.strip() for block in blocks for line in block.splitlines() if line.strip()]


def _tag_well_formed(text: str, tag: str) -> bool:
    opening = len(re.findall(rf"<{tag}>", text, flags=re.IGNORECASE))
    closing = len(re.findall(rf"</{tag}>", text, flags=re.IGNORECASE))
    complete = len(re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL))
    return opening == closing == complete


def _normalized_tokens(text: str) -> list[str]:
    # Keep target aggregation aligned with the evaluator's acceptable-answer
    # semantics while remaining self-contained so the same
    # estimator can run in both the legacy CUDA tree and the private VERL-0.7
    # NPU overlay.  Punctuation is separated (not concatenated), articles are
    # removed, then whitespace is canonicalized.
    text = str(text).lower()
    text = "".join(" " if character in set(string.punctuation) else character for character in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split()).split()


def _extract_solution(text: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", str(text), flags=re.IGNORECASE | re.DOTALL)
    if not matches:
        return None
    short = [match for match in matches if len(match) < 200]
    return (short[-1] if short else min(matches, key=len)).strip()


def _target_token_recall(text: str, target: str) -> float:
    target_tokens = _normalized_tokens(target)
    if not target_tokens:
        return 0.0
    text_counts = Counter(_normalized_tokens(text))
    target_counts = Counter(target_tokens)
    overlap = sum(min(count, text_counts.get(token, 0)) for token, count in target_counts.items())
    return float(overlap / len(target_tokens))


def best_acceptable_target_coverage(
    lines: Sequence[str],
    targets: Sequence[str],
    *,
    rank_discount: bool = False,
) -> float:
    """Best target-token recall in one evidence line.

    Evidence is evaluated line-by-line so common tokens spread over unrelated
    documents cannot be assembled into a spurious match.  Multiple targets are
    alternatives under the official evaluator, hence the maximum rather than a
    union or average.
    """

    best = 0.0
    for rank, line in enumerate(lines):
        discount = 1.0 / math.log2(rank + 2.0) if rank_discount else 1.0
        for target in targets:
            best = max(best, _target_token_recall(line, str(target)) * discount)
    return float(min(max(best, 0.0), 1.0))


def best_acceptable_answer_f1(prediction: str | None, targets: Sequence[str]) -> float:
    if prediction is None:
        return 0.0
    prediction_tokens = _normalized_tokens(prediction)
    if not prediction_tokens:
        return 0.0
    prediction_counts = Counter(prediction_tokens)
    best = 0.0
    for target in targets:
        target_tokens = _normalized_tokens(str(target))
        if not target_tokens:
            continue
        target_counts = Counter(target_tokens)
        overlap = sum(min(count, target_counts.get(token, 0)) for token, count in prediction_counts.items())
        if overlap <= 0:
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(target_tokens)
        best = max(best, 2.0 * precision * recall / (precision + recall))
    return float(min(max(best, 0.0), 1.0))


@dataclass(frozen=True)
class CoverageFrontierObservation:
    query_score: float
    evidence_score: float
    reasoning_score: float
    query_opportunity: float
    evidence_opportunity: float
    reasoning_opportunity: float
    query_valid: bool
    evidence_valid: bool
    reasoning_valid: bool
    raw_coverage: float
    selected_coverage: float
    answer_f1: float
    failure_reason: str = ""
    query_teacher_routable: bool = False
    evidence_teacher_routable: bool = False
    reasoning_teacher_routable: bool = False
    query_opportunity_valid: bool = False
    evidence_opportunity_valid: bool = False
    reasoning_opportunity_valid: bool = False
    rank_present: bool = False
    filter_present: bool = False
    query_failure_reason: str = ""
    evidence_failure_reason: str = ""
    reasoning_failure_reason: str = ""
    selected_line_count: int = 0
    provenance_line_count: int = 0
    selection_provenance_rate: float = 0.0
    target_injection_detected: bool = False


def _canonical_evidence_line(text: str) -> str:
    """Canonicalize formatting prefixes without using answer tokens."""

    text = _DOC_PREFIX_RE.sub("", str(text)).replace("Title:", " ").replace(")", " ")
    return " ".join(_normalized_tokens(text))


def _provenance_filter(
    selected_lines: Sequence[str],
    source_lines: Sequence[str],
) -> tuple[list[str], int]:
    """Keep only lines that exactly map to an upstream evidence fact.

    Matching deliberately does not use the target string, preventing circular
    validation where mentioning the answer is itself treated as provenance.
    """

    source = {_canonical_evidence_line(line) for line in source_lines}
    source.discard("")
    traced = [line for line in selected_lines if _canonical_evidence_line(line) in source]
    return traced, len(traced)


def compute_coverage_frontier_observation(
    response_text: str,
    targets: Sequence[str],
    *,
    parse_reliable: bool,
    search_transition_complete: bool,
    query_has_tokens: bool,
    evidence_has_tokens: bool,
    reasoning_has_tokens: bool,
    externally_truncated: bool = False,
    validity_mode: str = "strict_exclusion",
    parse_failure_reason: str = "",
    stage_parse_reliable: dict[str, bool] | None = None,
    enforce_selection_provenance: bool = True,
) -> CoverageFrontierObservation:
    """Measure observable progress without changing the task reward."""

    targets = [str(target) for target in targets if str(target).strip()]
    if validity_mode not in CF_VALIDITY_MODES:
        raise ValueError(f"Unsupported CF validity_mode={validity_mode!r}")
    if validity_mode == "strict_exclusion" and externally_truncated:
        return CoverageFrontierObservation(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, False, 0.0, 0.0, 0.0,
            "external_truncation",
        )
    if validity_mode == "strict_exclusion" and not parse_reliable:
        return CoverageFrontierObservation(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, False, 0.0, 0.0, 0.0,
            "span_parse_unreliable",
        )
    if not targets:
        return CoverageFrontierObservation(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, False, 0.0, 0.0, 0.0,
            "missing_target",
        )

    stage_parse_reliable = stage_parse_reliable or {}
    globally_unalignable = parse_failure_reason in {
        "token_alignment_failed",
        "stage_mask_overlap",
        "stage_mask_outside_actor_mask",
        "information_stage_overlap",
    }
    if validity_mode == "stage_local_valid" and globally_unalignable:
        return CoverageFrontierObservation(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, False, 0.0, 0.0, 0.0,
            parse_failure_reason,
        )

    information_lines = _tag_lines(response_text, "information")
    filter_lines = _tag_lines(response_text, "filter", last_only=True)
    rank_lines = _tag_lines(response_text, "rank", last_only=True)
    rank_present = bool(rank_lines)
    filter_present = bool(filter_lines)
    query_prefix_reliable = bool(
        stage_parse_reliable.get("query", stage_parse_reliable.get("search", parse_reliable))
        and _tag_well_formed(response_text, "search")
        and _tag_well_formed(response_text, "information")
        and search_transition_complete
    )
    selection_syntax_reliable = bool(
        stage_parse_reliable.get("evidence", stage_parse_reliable.get("selection", parse_reliable))
        and _tag_well_formed(response_text, "filter")
        and _tag_well_formed(response_text, "rank")
    )
    reasoning_syntax_reliable = bool(
        stage_parse_reliable.get("reasoning", parse_reliable)
        and _tag_well_formed(response_text, "answer")
        and bool(_tag_lines(response_text, "answer"))
    )

    raw_coverage = best_acceptable_target_coverage(information_lines, targets) if query_prefix_reliable else 0.0
    selected_lines = rank_lines if rank_lines else filter_lines
    traced_filter_lines, _ = _provenance_filter(filter_lines, information_lines)
    provenance_sources = [*information_lines, *traced_filter_lines]
    traced_selected_lines, provenance_count = _provenance_filter(selected_lines, provenance_sources)
    scoring_lines = traced_selected_lines if enforce_selection_provenance else selected_lines
    # Rank and no-rank paths now share the same positional scale.  Omitting
    # <rank> can no longer bypass the rank discount.
    selected_coverage = (
        best_acceptable_target_coverage(scoring_lines, targets, rank_discount=True)
        if selection_syntax_reliable
        else 0.0
    )
    selected_line_count = len(selected_lines)
    provenance_rate = provenance_count / selected_line_count if selected_line_count else 0.0
    untraced = [line for line in selected_lines if line not in traced_selected_lines]
    target_injection = best_acceptable_target_coverage(untraced, targets) > 0.0

    # Retention is conditional on what retrieval made available.  The absolute
    # selected coverage is retained separately as the reasoning opportunity.
    evidence_score = min(selected_coverage / raw_coverage, 1.0) if raw_coverage > _EPS else 0.0
    prediction = _extract_solution(response_text) if reasoning_syntax_reliable else None
    answer_f1 = best_acceptable_answer_f1(prediction, targets)

    # A completed search->information transition is evaluable even when the
    # retriever returned an empty block.  Empty retrieval is a valid score=0,
    # not missing supervision.
    if validity_mode == "stage_local_valid":
        # A completed, non-truncated response that omits or damages Search is
        # an observable model-controlled zero-score action.  It participates
        # in the sibling baseline but cannot become a teacher without a safe
        # Query span.  If generation was externally truncated before Search
        # became observable, there is no such attribution.
        query_valid = bool(query_prefix_reliable or not externally_truncated)
    else:
        # Empty but well-formed <search></search> is a valid zero-score
        # sibling.  ``query_has_tokens`` controls teacher routing, not whether
        # the failure may establish the comparison baseline.
        query_valid = bool(query_prefix_reliable)
    # A missing/empty selection is a model-controlled failure (score 0), not an
    # invalid example.  It remains in sibling comparisons, while its empty mask
    # prevents it from becoming a positive teacher itself.
    selection_closed_before_truncation = bool(
        selection_syntax_reliable and (filter_present or rank_present)
    )
    evidence_valid = bool(
        query_valid
        and raw_coverage > _EPS
        and (not externally_truncated or selection_closed_before_truncation)
    )
    if validity_mode == "stage_local_valid":
        # Missing/malformed answers are observable score-zero failures.  They
        # participate in the sibling baseline but cannot receive positive
        # credit unless a reliable reasoning mask exists.
        reasoning_valid = bool(
            evidence_valid and selected_coverage > _EPS and not externally_truncated
        )
    else:
        reasoning_valid = bool(
            evidence_valid and selected_coverage > _EPS and reasoning_has_tokens and prediction is not None
        )

    query_teacher_routable = bool(
        query_has_tokens
        and stage_parse_reliable.get("query", stage_parse_reliable.get("search", parse_reliable))
    )
    evidence_teacher_routable = bool(
        evidence_has_tokens
        and stage_parse_reliable.get("evidence", stage_parse_reliable.get("selection", parse_reliable))
    )
    reasoning_teacher_routable = bool(
        reasoning_has_tokens and stage_parse_reliable.get("reasoning", parse_reliable)
    )
    if query_prefix_reliable:
        query_failure = ""
    elif query_valid:
        query_failure = "model_controlled_search_missing_or_malformed"
    elif externally_truncated:
        query_failure = "external_truncation_before_search_closed"
    else:
        query_failure = parse_failure_reason or "search_transition_unavailable"
    evidence_failure = ""
    if evidence_valid and not selection_syntax_reliable:
        evidence_failure = "model_controlled_selection_malformed"
    elif externally_truncated and not selection_closed_before_truncation:
        evidence_failure = "external_truncation_before_selection_closed"
    elif not evidence_valid:
        evidence_failure = "no_evidence_opportunity"
    reasoning_failure = ""
    if reasoning_valid and prediction is None:
        reasoning_failure = "model_controlled_answer_missing_or_malformed"
    elif externally_truncated:
        reasoning_failure = "external_truncation_before_reasoning_closed"
    elif not reasoning_valid:
        reasoning_failure = "no_reasoning_opportunity"

    return CoverageFrontierObservation(
        query_score=float(raw_coverage),
        evidence_score=float(evidence_score),
        reasoning_score=float(answer_f1),
        query_opportunity=1.0,
        evidence_opportunity=float(raw_coverage),
        reasoning_opportunity=float(selected_coverage),
        query_valid=query_valid,
        evidence_valid=evidence_valid,
        reasoning_valid=reasoning_valid,
        raw_coverage=float(raw_coverage),
        selected_coverage=float(selected_coverage),
        answer_f1=float(answer_f1),
        failure_reason=query_failure,
        query_teacher_routable=query_teacher_routable,
        evidence_teacher_routable=evidence_teacher_routable,
        reasoning_teacher_routable=reasoning_teacher_routable,
        query_opportunity_valid=query_valid,
        evidence_opportunity_valid=evidence_valid,
        reasoning_opportunity_valid=reasoning_valid,
        rank_present=rank_present,
        filter_present=filter_present,
        query_failure_reason=query_failure,
        evidence_failure_reason=evidence_failure,
        reasoning_failure_reason=reasoning_failure,
        selected_line_count=selected_line_count,
        provenance_line_count=provenance_count,
        selection_provenance_rate=float(provenance_rate),
        target_injection_detected=bool(target_injection),
    )


@dataclass
class CoverageFrontierBatchCredit:
    scores: dict[str, torch.Tensor]
    opportunities: dict[str, torch.Tensor]
    valid: dict[str, torch.Tensor]
    frontier: dict[str, torch.Tensor]
    advantages: dict[str, torch.Tensor]
    stage_weights: dict[str, torch.Tensor]
    sequence_weights: dict[str, torch.Tensor]
    learnability: dict[str, torch.Tensor]
    dispersion: dict[str, torch.Tensor]
    top_margin: dict[str, torch.Tensor]
    teacher_routable: dict[str, torch.Tensor]
    group_learnability_sum: torch.Tensor
    group_complete: torch.Tensor
    group_active: torch.Tensor
    group_all_failure: torch.Tensor
    failure_reasons: list[str]


def build_coverage_frontier_token_routing(
    *,
    actor_loss_mask: torch.Tensor,
    stage_masks: dict[str, torch.Tensor],
    credit: CoverageFrontierBatchCredit,
) -> dict[str, torch.Tensor]:
    """Expand sequence credit into a length-invariant token objective.

    Every stage's token weights integrate to its driver-side sequence weight.
    This preserves the per-question stage-routing coefficient while preventing
    a long Selection span from receiving more total auxiliary mass solely by
    length.
    """

    actor = torch.as_tensor(actor_loss_mask, dtype=torch.float32)
    if actor.ndim != 2:
        raise ValueError("actor_loss_mask must have shape [batch, response_length]")
    local_advantage = torch.zeros_like(actor)
    local_weight = torch.zeros_like(actor)
    routed_masks: dict[str, torch.Tensor] = {}
    stage_union = torch.zeros_like(actor)
    for stage in CF_STAGES:
        if stage not in stage_masks:
            raise KeyError(f"Missing CF stage mask: {stage}")
        mask = torch.as_tensor(stage_masks[stage], dtype=actor.dtype)
        if mask.shape != actor.shape:
            raise ValueError(f"CF stage mask shape mismatch for {stage}")
        if (mask * (1.0 - actor)).abs().sum().item() != 0:
            raise ValueError(f"CF {stage} mask extends outside actor_loss_mask")
        if (stage_union * mask).abs().sum().item() != 0:
            raise ValueError("CF stage masks overlap")
        stage_union = stage_union + mask
        token_count = mask.sum(dim=-1).clamp_min(1.0)
        advantage = credit.advantages[stage].to(actor.dtype)
        sequence_weight = credit.sequence_weights[stage].to(actor.dtype)
        if advantage.shape != actor.shape[:1] or sequence_weight.shape != actor.shape[:1]:
            raise ValueError(f"CF sequence tensor shape mismatch for {stage}")
        local_advantage = local_advantage + advantage.unsqueeze(-1) * mask
        local_weight = local_weight + (sequence_weight / token_count).unsqueeze(-1) * mask
        routed_masks[f"cf_{stage}_mask"] = mask
    if not torch.isfinite(local_advantage).all() or not torch.isfinite(local_weight).all():
        raise ValueError("CF routed tensors contain NaN or Inf")
    if (local_weight < 0).any():
        raise ValueError("CF routed policy weights must be non-negative")
    return {
        "cf_local_advantage": local_advantage,
        "cf_local_weight": local_weight,
        **routed_masks,
    }


def _pairwise_frontier(scores: torch.Tensor, valid: torch.Tensor, delta: float) -> torch.Tensor:
    result = torch.zeros_like(scores)
    valid_indices = torch.nonzero(valid > 0, as_tuple=False).flatten()
    if valid_indices.numel() < 2:
        return result
    values = scores[valid_indices]
    diff = values[:, None] - values[None, :]
    wins = (diff > delta).to(values.dtype)
    losses = (diff < -delta).to(values.dtype)
    # Excluding the diagonal leaves exactly n-1 sibling comparisons.
    result[valid_indices] = (wins - losses).sum(dim=1) / float(values.numel() - 1)
    return result


def _mean_pairwise_distance(values: torch.Tensor) -> torch.Tensor:
    if values.numel() < 2:
        return values.new_zeros(())
    diff = (values[:, None] - values[None, :]).abs()
    return diff.sum() / float(values.numel() * (values.numel() - 1))


def compute_coverage_frontier_batch_credit(
    *,
    uids: Sequence[Any],
    observations: Sequence[CoverageFrontierObservation],
    outcome_rewards: Sequence[float] | torch.Tensor,
    stage_has_tokens: dict[str, Sequence[bool] | torch.Tensor],
    expected_group_size: int = 5,
    require_complete_group: bool = True,
    min_comparable: int = 2,
    pairwise_delta: float = 0.0,
    difficulty_weight: str = "none",
) -> CoverageFrontierBatchCredit:
    """Compute all sibling statistics before Ray/DP/micro-batch splitting."""

    n = len(observations)
    if len(uids) != n:
        raise ValueError("uids and observations must have identical lengths")
    outcome = torch.as_tensor(outcome_rewards, dtype=torch.float32).flatten()
    if outcome.numel() != n:
        raise ValueError("outcome_rewards must have one value per rollout")
    if not torch.isfinite(outcome).all():
        raise ValueError("outcome_rewards contains NaN or Inf")
    if difficulty_weight not in {"none", "one_minus_success_rate"}:
        raise ValueError(f"Unsupported difficulty_weight={difficulty_weight!r}")

    def values(name: str, field: str) -> torch.Tensor:
        return torch.tensor([float(getattr(obs, f"{name}_{field}")) for obs in observations], dtype=torch.float32)

    scores = {stage: values(stage, "score") for stage in CF_STAGES}
    opportunities = {stage: values(stage, "opportunity") for stage in CF_STAGES}
    valid = {
        stage: torch.tensor([float(getattr(obs, f"{stage}_valid")) for obs in observations], dtype=torch.float32)
        for stage in CF_STAGES
    }
    token_flags = {
        stage: torch.as_tensor(stage_has_tokens[stage], dtype=torch.float32).flatten() for stage in CF_STAGES
    }
    for stage in CF_STAGES:
        if token_flags[stage].numel() != n:
            raise ValueError(f"stage_has_tokens[{stage!r}] has the wrong length")
        if not torch.isfinite(scores[stage]).all() or not torch.isfinite(opportunities[stage]).all():
            raise ValueError(f"Non-finite CF score/opportunity for {stage}")

    frontier = {stage: torch.zeros(n, dtype=torch.float32) for stage in CF_STAGES}
    advantages = {stage: torch.zeros(n, dtype=torch.float32) for stage in CF_STAGES}
    stage_weights = {stage: torch.zeros(n, dtype=torch.float32) for stage in CF_STAGES}
    sequence_weights = {stage: torch.zeros(n, dtype=torch.float32) for stage in CF_STAGES}
    learnability = {stage: torch.zeros(n, dtype=torch.float32) for stage in CF_STAGES}
    dispersion_by_stage = {stage: torch.zeros(n, dtype=torch.float32) for stage in CF_STAGES}
    top_margin = {stage: torch.zeros(n, dtype=torch.float32) for stage in CF_STAGES}
    group_complete = torch.zeros(n, dtype=torch.float32)
    group_active = torch.zeros(n, dtype=torch.float32)
    group_all_failure = torch.zeros(n, dtype=torch.float32)
    group_learnability_sum = torch.zeros(n, dtype=torch.float32)

    groups: dict[tuple[str, Any], list[int]] = {}
    for index, uid in enumerate(uids):
        groups.setdefault(_uid_key(uid), []).append(index)

    for indices in groups.values():
        index = torch.tensor(indices, dtype=torch.long)
        complete = len(indices) == int(expected_group_size)
        group_complete[index] = float(complete)
        all_failure = bool((outcome[index] <= 0).all())
        group_all_failure[index] = float(all_failure)
        if require_complete_group and not complete:
            continue

        ell: dict[str, torch.Tensor] = {}
        for stage in CF_STAGES:
            stage_valid = valid[stage][index]
            valid_local = torch.nonzero(stage_valid > 0, as_tuple=False).flatten()
            if valid_local.numel() < int(min_comparable):
                ell[stage] = torch.tensor(0.0)
                continue
            local_frontier = _pairwise_frontier(scores[stage][index], stage_valid, float(pairwise_delta))
            frontier[stage][index] = local_frontier
            local_advantage = (
                opportunities[stage][index]
                * scores[stage][index]
                * torch.clamp(local_frontier, min=0.0)
                * stage_valid
                * token_flags[stage][index]
            )
            advantages[stage][index] = local_advantage

            valid_scores = scores[stage][index][valid_local]
            valid_opportunity = opportunities[stage][index][valid_local]
            support = valid_local.numel() / float(len(indices))
            opportunity_mean = valid_opportunity.mean()
            dispersion = _mean_pairwise_distance(valid_scores)
            dispersion_by_stage[stage][index] = dispersion
            sorted_scores = torch.sort(valid_scores, descending=True).values
            if sorted_scores.numel() >= 2:
                top_margin[stage][index] = sorted_scores[0] - sorted_scores[1]
            headroom = 1.0 - valid_scores.mean()
            stage_ell = support * opportunity_mean * dispersion * torch.clamp(headroom, min=0.0)
            # A stage without a positive, token-bearing teacher cannot spend a
            # local budget in this group even if its failures are evaluable.
            if not (local_advantage > 0).any():
                stage_ell = stage_ell.new_zeros(())
            ell[stage] = stage_ell
            learnability[stage][index] = stage_ell

        ell_sum = sum(ell.values())
        if not torch.isfinite(ell_sum) or ell_sum <= _EPS:
            continue
        group_learnability_sum[index] = ell_sum
        group_active[index] = 1.0
        difficulty = 1.0
        if difficulty_weight == "one_minus_success_rate":
            difficulty = float(1.0 - outcome[index].mean().item())
        for stage in CF_STAGES:
            weight = float((ell[stage] / ell_sum).item()) if ell[stage] > 0 else 0.0
            stage_weights[stage][index] = weight
            teachers = (advantages[stage][index] > 0) & (token_flags[stage][index] > 0)
            teacher_count = int(teachers.sum().item())
            if teacher_count > 0 and weight > 0:
                selected = index[teachers]
                sequence_weights[stage][selected] = difficulty * weight / float(teacher_count)

    for collection in (
        scores, opportunities, valid, frontier, advantages, stage_weights,
        sequence_weights, learnability, dispersion_by_stage, top_margin,
    ):
        for tensor in collection.values():
            if not torch.isfinite(tensor).all():
                raise ValueError("CF credit contains NaN or Inf")
    return CoverageFrontierBatchCredit(
        scores=scores,
        opportunities=opportunities,
        valid=valid,
        frontier=frontier,
        advantages=advantages,
        stage_weights=stage_weights,
        sequence_weights=sequence_weights,
        learnability=learnability,
        dispersion=dispersion_by_stage,
        top_margin=top_margin,
        teacher_routable=token_flags,
        group_learnability_sum=group_learnability_sum,
        group_complete=group_complete,
        group_active=group_active,
        group_all_failure=group_all_failure,
        failure_reasons=[obs.failure_reason for obs in observations],
    )


def compute_opportunity_reversal_diagnostics(
    *,
    uids: Sequence[Any],
    observations: Sequence[CoverageFrontierObservation],
    pairwise_delta: float = 0.0,
) -> dict[str, float]:
    """Measure when conditional-quality and effective-progress order disagree.

    Besides raw pair reversals, report whether the alternative ordering would
    actually change the positive-teacher set and how much of the current local
    credit mass lies on affected teachers.  These are diagnostics only; they
    never alter the CF objective.
    """

    groups: dict[tuple[str, Any], list[int]] = {}
    for index, uid in enumerate(uids):
        groups.setdefault(_uid_key(uid), []).append(index)
    counts = {
        "evidence_pairs": 0.0,
        "evidence_reverse": 0.0,
        "reasoning_pairs": 0.0,
        "reasoning_reverse": 0.0,
    }
    teacher_counts = {
        stage: {
            "compared": 0.0,
            "changed": 0.0,
            "current_mass": 0.0,
            "affected_current_mass": 0.0,
        }
        for stage in ("evidence", "reasoning")
    }
    for indices in groups.values():
        for stage, effective in (
            ("evidence", lambda obs: obs.selected_coverage),
            ("reasoning", lambda obs: obs.selected_coverage * obs.reasoning_score),
        ):
            valid_indices = [index for index in indices if getattr(observations[index], f"{stage}_valid")]
            if len(valid_indices) < 2:
                continue
            quality = torch.tensor(
                [getattr(observations[index], f"{stage}_score") for index in valid_indices],
                dtype=torch.float32,
            )
            progress = torch.tensor(
                [effective(observations[index]) for index in valid_indices],
                dtype=torch.float32,
            )
            all_valid = torch.ones(len(valid_indices), dtype=torch.float32)
            quality_frontier = _pairwise_frontier(quality, all_valid, pairwise_delta)
            progress_frontier = _pairwise_frontier(progress, all_valid, pairwise_delta)
            for local_index, rollout_index in enumerate(valid_indices):
                obs = observations[rollout_index]
                routable = bool(getattr(obs, f"{stage}_teacher_routable"))
                amplitude = float(
                    getattr(obs, f"{stage}_opportunity")
                    * getattr(obs, f"{stage}_score")
                )
                current_mass = amplitude * max(float(quality_frontier[local_index]), 0.0) if routable else 0.0
                alternative_mass = (
                    amplitude * max(float(progress_frontier[local_index]), 0.0) if routable else 0.0
                )
                changed = (current_mass > 0.0) != (alternative_mass > 0.0)
                teacher_counts[stage]["compared"] += 1.0
                teacher_counts[stage]["changed"] += float(changed)
                teacher_counts[stage]["current_mass"] += current_mass
                if changed:
                    teacher_counts[stage]["affected_current_mass"] += current_mass

        for left_pos, left in enumerate(indices):
            for right in indices[left_pos + 1 :]:
                for stage, effective in (
                    ("evidence", lambda obs: obs.selected_coverage),
                    ("reasoning", lambda obs: obs.selected_coverage * obs.reasoning_score),
                ):
                    lobs, robs = observations[left], observations[right]
                    if not getattr(lobs, f"{stage}_valid") or not getattr(robs, f"{stage}_valid"):
                        continue
                    quality_diff = getattr(lobs, f"{stage}_score") - getattr(robs, f"{stage}_score")
                    effective_diff = effective(lobs) - effective(robs)
                    if abs(quality_diff) <= pairwise_delta or abs(effective_diff) <= pairwise_delta:
                        continue
                    counts[f"{stage}_pairs"] += 1.0
                    if quality_diff * effective_diff < 0:
                        counts[f"{stage}_reverse"] += 1.0
    result = {
        "evidence_pair_count": counts["evidence_pairs"],
        "evidence_reverse_count": counts["evidence_reverse"],
        "evidence_reverse_rate": counts["evidence_reverse"] / max(counts["evidence_pairs"], 1.0),
        "reasoning_pair_count": counts["reasoning_pairs"],
        "reasoning_reverse_count": counts["reasoning_reverse"],
        "reasoning_reverse_rate": counts["reasoning_reverse"] / max(counts["reasoning_pairs"], 1.0),
    }
    for stage in ("evidence", "reasoning"):
        item = teacher_counts[stage]
        result[f"{stage}_teacher_status_comparison_count"] = item["compared"]
        result[f"{stage}_teacher_set_change_count"] = item["changed"]
        result[f"{stage}_teacher_set_change_rate"] = item["changed"] / max(item["compared"], 1.0)
        result[f"{stage}_affected_current_credit_mass_rate"] = item[
            "affected_current_mass"
        ] / max(item["current_mass"], _EPS)
    return result
