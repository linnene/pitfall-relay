"""Packaging smoke checks, not scientific validation."""

import unittest

import pitfall_relay


class PackageTest(unittest.TestCase):
    def test_import_and_version(self):
        self.assertEqual(pitfall_relay.__version__, "0.0.1")
