import unittest

import failure_gate as f


class FailureGateTest(unittest.TestCase):
    def test_order_causes_failure_not_skill(self):
        for seed in (6001, 6002):
            pair = f.make_pair(seed, "similar")
            self.assertTrue(pair["a"]["pump_first"]["lost"])
            self.assertTrue(pair["a"]["transport_first"]["preserved"])

    def test_normal_recovers_attempt(self):
        pair = f.make_pair(6001, "similar")
        nv = f.normal_view(pair)
        ev = f.experience_view(pair)
        self.assertEqual(nv["observed_order"], ev["attempt"]["order"])
        self.assertEqual(nv["a_failure_notice"], ev["outcome"]["lost"])

    def test_normal_and_experience_choice_same(self):
        pair = f.make_pair(6001, "similar")
        nv = f.normal_view(pair)
        ev = f.experience_view(pair)
        self.assertEqual(f._normal_choice(nv, pair["b"]),
                         f._experience_choice(ev, pair["b"]))

    def test_dev_is_deterministic_and_all_arms_same(self):
        a = f.run_dev(2)
        b = f.run_dev(2)
        self.assertEqual(a, b)
        for kind in ("similar", "nonapp"):
            rows = [x for x in a["rows"] if x["kind"] == kind]
            by_seed = {}
            for x in rows:
                by_seed.setdefault(x["seed"], set()).add((x["choice"], x["b_lost"]))
            self.assertTrue(all(len(v) == 1 for v in by_seed.values()))


if __name__ == "__main__":
    unittest.main()
