import unittest

import rescue_mechanism as m


class RescueMechanismTest(unittest.TestCase):
    def test_reversal_both_regimes(self):
        r = m.precheck(2)
        for regime in ("parallel_better", "serial_better"):
            rows = [x for x in r["rows"] if x["regime"] == regime]
            self.assertTrue(all(x["reversal"] for x in rows))
            if regime == "parallel_better":
                self.assertTrue(all(x["parallel"]["team"] > x["serial"]["team"] for x in rows))
            else:
                self.assertTrue(all(x["serial"]["team"] > x["parallel"]["team"] for x in rows))

    def test_strong_rule_matches_experience(self):
        for seed in (8001, 8101):
            case = m.make_case(seed, "parallel_better" if seed < 8100 else "serial_better")
            s = m.simulate(case, "serial")
            p = m.simulate(case, "parallel")
            exp = m.experience_rule(case, "serial", s["team"], p["team"])
            self.assertEqual(exp, m.strong_rule(case))

    def test_no_invalid_completion(self):
        case = m.make_case(8001, "parallel_better")
        for plan in ("serial", "parallel"):
            out = m.simulate(case, plan)
            self.assertIn(out["team"], (0, 1, 2))
            self.assertIn("a_ok", out)
            self.assertIn("b_ok", out)

    def test_deterministic(self):
        self.assertEqual(m.run_dev(), m.run_dev())


if __name__ == "__main__":
    unittest.main()
