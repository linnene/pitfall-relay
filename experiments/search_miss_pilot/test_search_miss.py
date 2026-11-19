import unittest

import search_miss as s


class SearchMissTest(unittest.TestCase):
    def test_a_fast_miss_and_slow_find(self):
        for kind in ("relevant", "nonapp"):
            seed = 7001 if kind == "relevant" else 7101
            a = s.make_chamber(seed, hidden=(kind == "relevant"))
            fast = s.simulate_a(a, "fast")
            slow = s.simulate_a(a, "slow")
            if kind == "relevant":
                self.assertTrue(fast["miss"])
            else:
                self.assertFalse(fast["miss"])
            self.assertTrue(slow["found"])

    def test_normal_and_experience_choice_same(self):
        a = s.make_chamber(7001, hidden=True)
        b = s.make_chamber(7051, hidden=True)
        nv = s.normal_view(a, b)
        ev = s.experience_view(a, s.simulate_a(a, "fast"), s.simulate_a(a, "slow"))
        self.assertEqual(s.strong_coverage_choice(nv), s.experience_choice(ev, nv))
        self.assertEqual(s.strong_coverage_choice(nv), "slow")

    def test_nonapp_fast_and_permuted_slow(self):
        a = s.make_chamber(7101, hidden=False)
        b = s.make_chamber(7151, hidden=False)
        nv = s.normal_view(a, b)
        self.assertEqual(s.strong_coverage_choice(nv), "fast")

    def test_deterministic(self):
        self.assertEqual(s.run_dev(2), s.run_dev(2))


if __name__ == "__main__":
    unittest.main()
