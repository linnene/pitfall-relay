import unittest

import flood_recheck as f


class FloodRecheckTest(unittest.TestCase):
    def test_dev_categories_valid(self):
        r = f.run_groups({c: [5001 + 100 * i] for i, c in enumerate(f.CATEGORIES)})
        self.assertEqual(sum(s["errors"] for s in r["summary"].values()), 0)
        for c in f.CATEGORIES:
            self.assertEqual(r["summary"][c]["n"], 1)

    def test_same_seed_reproducible_and_arms_share_scenario(self):
        a = f.one(5001, "relevant_pump")
        b = f.one(5001, "relevant_pump")
        self.assertEqual(a["a_sensor"], b["a_sensor"])
        self.assertEqual(a["a_exec"], b["a_exec"])
        self.assertEqual(a["t0"], b["t0"])
        self.assertEqual(a["a_lost"], b["a_lost"])

    def test_nonnegative_water(self):
        scn = f.make_scenario(5001, "relevant_pump")
        out = f.base._sim_area(scn.a, scn.shared, scn.local_a,
                               "pump_first", scn.horizon)
        self.assertTrue(all(x >= 0.0 for x in out["level"]))


if __name__ == "__main__":
    unittest.main()
