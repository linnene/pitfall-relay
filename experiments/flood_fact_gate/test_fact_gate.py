import unittest

import fact_gate as f


class FactGateTest(unittest.TestCase):
    def test_pair_normal_trace_and_opposite_fact(self):
        a = f.scenario(4001, "relevant")
        b = f.scenario(4101, "nonapp")
        self.assertEqual(a["a_levels"], b["a_levels"])
        self.assertEqual(a["b_levels"], b["b_levels"])
        self.assertNotEqual(a["a_pump"], b["a_pump"])
        fa = f._fact_input(a)
        fb = f._fact_input(b)
        self.assertGreater(f._estimate_qs(fa), 0.6)
        self.assertLess(f._estimate_qs(fb), 0.6)

    def test_baseline_cannot_see_fact(self):
        r = f.run({"relevant": [4001]}, None)
        for row in r["rows"]:
            if row["kind"] == "notimely" or row["arm"] == "baseline":
                self.assertFalse(row["fact_visible"])
            else:
                self.assertTrue(row["fact_visible"])

    def test_notimely_arms_identical(self):
        r = f.run({"notimely": [4201]}, None)
        rows = [x for x in r["rows"] if x["seed"] == 4201]
        self.assertEqual(len({(x["action"], x["b_lost"]) for x in rows}), 1)

    def test_deterministic_and_nonnegative(self):
        a = f.scenario(4001, "relevant")
        self.assertTrue(all(x >= 0 for x in a["a_levels"]))
        r1 = f.run({"relevant": [4001]}, None)
        r2 = f.run({"relevant": [4001]}, None)
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
