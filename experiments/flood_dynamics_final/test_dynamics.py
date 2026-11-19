import unittest

import dynamics as d


class DynamicsTest(unittest.TestCase):
    def test_same_normal_trace_different_fact(self):
        a = d.scenario(4001, "relevant")
        b = d.scenario(4101, "nonapp")
        self.assertEqual(a["a_levels"], b["a_levels"])
        self.assertEqual(a["b_levels"], b["b_levels"])
        self.assertEqual(a["b_damage"], b["b_damage"])
        self.assertNotEqual(a["a_p"], b["a_p"])
        self.assertGreater(d.estimate_qs(d.fact_input(a)), 0.06)
        self.assertLess(d.estimate_qs(d.fact_input(b)), 0.06)

    def test_baseline_has_no_fact(self):
        r = d.run({"relevant": [4001]}, None)
        for row in r["rows"]:
            if row["kind"] == "notimely" or row["arm"] == "baseline":
                self.assertFalse(row["fact_visible"])
            else:
                self.assertTrue(row["fact_visible"])

    def test_notimely_equal_and_nonnegative(self):
        r = d.run({"notimely": [4201]}, None)
        rows = [x for x in r["rows"] if x["seed"] == 4201]
        self.assertEqual(len({(x["action"], x["b_lost"]) for x in rows}), 1)
        scn = d.scenario(4001, "relevant")
        self.assertTrue(all(x >= 0 for x in scn["a_levels"]))
        self.assertTrue(all(x >= 0 for x in scn["b_levels"]))

    def test_deterministic(self):
        self.assertEqual(d.run({"relevant": [4001]}, None),
                         d.run({"relevant": [4001]}, None))


if __name__ == "__main__":
    unittest.main()
