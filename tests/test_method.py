"""Synthetic contracts; no benchmark assets are required."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'verl/trainer/ppo'))

import torch
import coverage_frontier_psr as cf
from ocor_runtime import build_runtime_weights
from search_r1.reward_fn_binary import compute_score


class MethodTests(unittest.TestCase):
    def test_terminal_reward_is_substring(self):
        self.assertEqual(compute_score('TASK', '<answer>Entity B and Entity C</answer>',
                                      {'target': ['Entity B']}), 1.0)
        self.assertEqual(compute_score('TASK', '<answer>Entity C</answer>',
                                      {'target': ['Entity B']}), 0.0)
        self.assertEqual(compute_score('TASK', 'No answer block', {'target': ['Entity B']}), 0.0)

    def test_local_target_semantics(self):
        self.assertEqual(cf.best_acceptable_target_coverage(['Entity B'], ['Entity B', 'Entity C']), 1.0)
        self.assertEqual(cf.best_acceptable_answer_f1('Entity B', ['Entity B']), 1.0)
        self.assertLess(cf.best_acceptable_answer_f1('Entity B and Entity C', ['Entity B']), 1.0)

    def test_provenance_rejects_invented_fact(self):
        kept, count = cf._provenance_filter(['Entity B holds Office on Year X'],
                                          ['Entity C holds Office on Year X'])
        self.assertEqual(kept, [])
        self.assertEqual(count, 0)

    def test_mass_conservation_and_gate(self):
        actor = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]], dtype=torch.float32)
        realization = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]], dtype=torch.float32)
        result = build_runtime_weights(actor_mask=actor, realization_mask=realization,
            cf_reasoning_valid=torch.ones(2), cf_reasoning_opportunity=torch.tensor([1., 0.]),
            advantages=actor.clone(), uids=['q', 'q'], qlabels=['TASK'] * 2,
            qtypes=['TYPE'] * 2, arm='ocor', rho=0.04)
        torch.testing.assert_close(result.weights.sum(-1), actor.sum(-1))
        self.assertEqual(result.eligible.tolist(), [True, False])
        self.assertEqual(float(result.weights[:, -1].sum()), 0.0)
        self.assertGreater(float(result.weights[0, 2]), 1.0)
        torch.testing.assert_close(result.weights[1], actor[1])

    def test_invalid_mask_fails_closed(self):
        with self.assertRaises(ValueError):
            build_runtime_weights(actor_mask=torch.tensor([[1., 0.]]),
                realization_mask=torch.tensor([[0., 1.]]), cf_reasoning_valid=torch.ones(1),
                cf_reasoning_opportunity=torch.ones(1), advantages=torch.ones(1, 2),
                uids=['q'], qlabels=['TASK'], qtypes=['TYPE'], arm='ocor')

    def test_all_wrong_group_can_have_local_credit(self):
        observations = [cf.CoverageFrontierObservation(
            query_score=q, evidence_score=0., reasoning_score=0.,
            query_opportunity=1., evidence_opportunity=0., reasoning_opportunity=0.,
            query_valid=True, evidence_valid=True, reasoning_valid=True,
            raw_coverage=q, selected_coverage=0., answer_f1=0.) for q in (1., 0.)]
        rewards = torch.zeros(2)
        result = cf.compute_coverage_frontier_batch_credit(
            uids=['question', 'question'], observations=observations,
            outcome_rewards=rewards, expected_group_size=2,
            stage_has_tokens={s: [True, True] for s in cf.CF_STAGES})
        self.assertTrue(bool(result.group_all_failure.all()))
        self.assertGreater(float(result.advantages['query'][0]), 0.)
        self.assertEqual(float(result.advantages['query'][1]), 0.)
        torch.testing.assert_close(rewards, torch.zeros(2))

    def test_no_quality_difference_has_no_local_budget(self):
        observation = cf.CoverageFrontierObservation(
            query_score=0., evidence_score=0., reasoning_score=0.,
            query_opportunity=1., evidence_opportunity=0., reasoning_opportunity=0.,
            query_valid=True, evidence_valid=True, reasoning_valid=True,
            raw_coverage=0., selected_coverage=0., answer_f1=0.)
        result = cf.compute_coverage_frontier_batch_credit(
            uids=['question', 'question'], observations=[observation, observation],
            outcome_rewards=[0., 0.], expected_group_size=2,
            stage_has_tokens={s: [True, True] for s in cf.CF_STAGES})
        self.assertEqual(float(result.group_active.sum()), 0.)
        for weights in result.sequence_weights.values():
            self.assertEqual(float(weights.sum()), 0.)


if __name__ == '__main__':
    unittest.main()
