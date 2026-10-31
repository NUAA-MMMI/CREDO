"""Binary normalized-substring terminal reward, without format bonuses."""


def compute_score(data_source: str, solution_str: str, ground_truth, extra_info=None, **kwargs) -> float:
    del data_source, extra_info, kwargs
    from search_r1 import qa_em_format

    return float(
        qa_em_format.compute_score_em(
            solution_str=solution_str,
            ground_truth=ground_truth,
            structure_format_score=0.0,
            retrieval_score=0.0,
            final_format_score=0.0,
            score=1.0,
        )
    )

