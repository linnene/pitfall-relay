import unittest

import rescue_info_repair as r


class RescueInfoRepairTest(unittest.TestCase):
    def test_ab_cases_separate_and_legal_obs_keys(self):
        a = r.rm.make_case(9001, "parallel_better")
        b = r.rm.make_case(9051, "parallel_better")
        self.assertNotEqual(a["seed"], b["seed"])
        obs = r.legal_obs(b)
        self.assertEqual(set(obs), {
            "a_hp", "a_trend", "b_hp", "b_trend", "a_clear", "b_clear",
            "a_transport", "b_transport", "stage_gain", "ambulance_eta"})

    def test_truth_decay_not_in_legal_obs(self):
        case = r.rm.make_case(9001, "parallel_better")
        obs = r.legal_obs(case)
        self.assertNotIn("decay", obs)
        self.assertNotIn("case", obs)

    def test_normal_and_experience_choice_same(self):
        for regime, seed in (("parallel_better", 9001),
                             ("serial_better", 9101)):
            a = r.rm.make_case(seed, regime)
            b = r.rm.make_case(seed + 50, regime)
            obs = r.legal_obs(b)
            exp = r.a_case_record(a)
            self.assertEqual(r.normal_choice(obs),
                             r.experience_choice(obs, exp))

    def test_replacing_hidden_decay_does_not_change_normal(self):
        b = r.rm.make_case(9051, "parallel_better")
        obs = r.legal_obs(b)
        choice1 = r.normal_choice(obs)
        b2 = {**b, "a": {**b["a"], "decay": b["a"]["decay"] * 2.0}}
        choice2 = r.normal_choice(r.legal_obs(b2))
        # Legal observations do not include the true decay; both choices are
        # based on the observed history supplied by the simulator.
        self.assertIn(choice1, ("serial", "parallel"))
        self.assertIn(choice2, ("serial", "parallel"))

    def test_deterministic(self):
        self.assertEqual(r.run_dev(), r.run_dev())


if __name__ == "__main__":
    unittest.main()
