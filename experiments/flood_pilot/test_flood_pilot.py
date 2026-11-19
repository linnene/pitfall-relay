import unittest

import flood_pilot as f


class FloodPilotTest(unittest.TestCase):
    def test_gate1_reversal_and_repro(self):
        r = f.gate1(8)
        self.assertTrue(r["summary"]["high_a"]["all_repro"])
        self.assertTrue(r["summary"]["urgent_b"]["all_repro"])
        self.assertEqual(r["summary"]["high_a"]["transport_wins"], 8)
        self.assertEqual(r["summary"]["urgent_b"]["pump_wins"], 8)

    def test_nonnegative_water_under_oversized_pump(self):
        scn = f.make_gate1(1, "high_a")
        cfg = f.AreaCfg("A", 10.0, 1.0, 0.01, 1.0, 100.0, 0, 3, 1.0)
        out = f._sim_area(cfg, scn.shared, scn.local_a, "transport_first", 20)
        self.assertTrue(all(x >= 0.0 for x in out["level"]))
        self.assertTrue(all(x >= 0.0 for x in scn.shared))

    def test_gate2_arms_share_scenario(self):
        for seed, group in ((1, "relevant"), (1, "nonapplicable")):
            row = f._gate2_one(seed, group)
            self.assertEqual(row["seed"], seed)
            self.assertEqual(row["group"], group)
            self.assertIn("a_history", row)
            self.assertGreater(row["t0"], 0)


if __name__ == "__main__":
    unittest.main()
