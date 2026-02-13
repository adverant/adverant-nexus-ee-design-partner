"""
Verification tests for Component Search API Integrations.

Tests:
1. Part number normalization (LCSC suffix stripping)
2. MPN search variant generation (progressive truncation)
3. API method signatures exist and are callable
4. Live API calls (optional, requires credentials)

Run:
    python3 test_component_search.py           # Unit tests only
    python3 test_component_search.py --live     # Include live API tests
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import sys
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure the module is importable
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPartNumberNormalization(unittest.TestCase):
    """Test normalize_part_number() correctly splits MPN_LCSCID format."""

    def setUp(self):
        from symbol_assembler import normalize_part_number
        self.normalize = normalize_part_number

    def test_standard_lcsc_suffix(self):
        """C1005X7R1H104K_C523 -> ('C1005X7R1H104K', 'C523')"""
        mpn, lcsc_id = self.normalize("C1005X7R1H104K_C523")
        self.assertEqual(mpn, "C1005X7R1H104K")
        self.assertEqual(lcsc_id, "C523")

    def test_longer_lcsc_id(self):
        """STM32G431CBT6_C2652578 -> ('STM32G431CBT6', 'C2652578')"""
        mpn, lcsc_id = self.normalize("STM32G431CBT6_C2652578")
        self.assertEqual(mpn, "STM32G431CBT6")
        self.assertEqual(lcsc_id, "C2652578")

    def test_single_digit_lcsc(self):
        """LM7805_C1 -> ('LM7805', 'C1')"""
        mpn, lcsc_id = self.normalize("LM7805_C1")
        self.assertEqual(mpn, "LM7805")
        self.assertEqual(lcsc_id, "C1")

    def test_no_lcsc_suffix(self):
        """STM32G431CBT6 -> ('STM32G431CBT6', None)"""
        mpn, lcsc_id = self.normalize("STM32G431CBT6")
        self.assertEqual(mpn, "STM32G431CBT6")
        self.assertIsNone(lcsc_id)

    def test_non_lcsc_underscore_suffix(self):
        """RC0402FR-0710KL_RST -> ('RC0402FR-0710KL_RST', None)
        _RST is not an LCSC ID pattern (doesn't match C + digits)."""
        mpn, lcsc_id = self.normalize("RC0402FR-0710KL_RST")
        self.assertEqual(mpn, "RC0402FR-0710KL_RST")
        self.assertIsNone(lcsc_id)

    def test_empty_string(self):
        """Empty string -> ('', None)"""
        mpn, lcsc_id = self.normalize("")
        self.assertEqual(mpn, "")
        self.assertIsNone(lcsc_id)

    def test_just_underscore(self):
        """Bare underscore -> no match."""
        mpn, lcsc_id = self.normalize("_C123")
        # '_C123' matches the pattern with empty MPN group
        # The regex requires .+? (one or more chars) before underscore
        self.assertEqual(mpn, "_C123")
        self.assertIsNone(lcsc_id)

    def test_multiple_underscores_with_lcsc(self):
        """Part_SubPart_C999 -> ('Part_SubPart', 'C999')"""
        mpn, lcsc_id = self.normalize("Part_SubPart_C999")
        self.assertEqual(mpn, "Part_SubPart")
        self.assertEqual(lcsc_id, "C999")

    def test_lcsc_too_many_digits(self):
        """LCSC IDs have at most 7 digits. 8+ digits should not match."""
        mpn, lcsc_id = self.normalize("PART_C12345678")
        self.assertEqual(mpn, "PART_C12345678")
        self.assertIsNone(lcsc_id)


class TestMPNSearchVariants(unittest.TestCase):
    """Test generate_mpn_search_variants() produces correct truncations."""

    def setUp(self):
        from symbol_assembler import generate_mpn_search_variants
        self.gen = generate_mpn_search_variants

    def test_no_underscores(self):
        """No underscores -> single variant (original)."""
        self.assertEqual(self.gen("STM32G431CBT6"), ["STM32G431CBT6"])

    def test_single_underscore(self):
        """RC0402FR-0710KL_RST -> ['RC0402FR-0710KL_RST', 'RC0402FR-0710KL']"""
        variants = self.gen("RC0402FR-0710KL_RST")
        self.assertEqual(variants, ["RC0402FR-0710KL_RST", "RC0402FR-0710KL"])

    def test_multiple_underscores(self):
        """A_B_C -> ['A_B_C', 'A_B', 'A'] but 'A' is only 1 char, too short."""
        variants = self.gen("PART_SUB_C123")
        self.assertEqual(variants, ["PART_SUB_C123", "PART_SUB", "PART"])

    def test_short_fragments_excluded(self):
        """Fragments shorter than 4 chars are excluded."""
        variants = self.gen("AB_CD")
        # 'AB_CD' (5 chars) -> strip -> 'AB' (2 chars, excluded)
        self.assertEqual(variants, ["AB_CD"])

    def test_already_normalized(self):
        """Already clean MPN with no underscores."""
        variants = self.gen("C1005X7R1H104K")
        self.assertEqual(variants, ["C1005X7R1H104K"])

    def test_combined_normalize_then_variants(self):
        """Full pipeline: normalize strips LCSC, variants strip remaining."""
        from symbol_assembler import normalize_part_number
        raw = "RC0402FR-0710KL_RST_C523"
        mpn, lcsc_id = normalize_part_number(raw)
        # _C523 matches LCSC pattern -> stripped
        self.assertEqual(mpn, "RC0402FR-0710KL_RST")
        self.assertEqual(lcsc_id, "C523")
        # Now generate variants from the MPN
        variants = self.gen(mpn)
        self.assertEqual(variants, ["RC0402FR-0710KL_RST", "RC0402FR-0710KL"])


class TestAPIMethodSignatures(unittest.TestCase):
    """Verify new API methods exist with correct signatures."""

    def setUp(self):
        from symbol_assembler import SymbolAssembler
        self.cls = SymbolAssembler

    def test_search_digikey_api_exists(self):
        self.assertTrue(hasattr(self.cls, 'search_digikey_api'))
        sig = inspect.signature(self.cls.search_digikey_api)
        params = list(sig.parameters.keys())
        self.assertIn('mpn', params)

    def test_search_mouser_api_exists(self):
        self.assertTrue(hasattr(self.cls, 'search_mouser_api'))
        sig = inspect.signature(self.cls.search_mouser_api)
        params = list(sig.parameters.keys())
        self.assertIn('mpn', params)

    def test_search_lcsc_api_exists(self):
        self.assertTrue(hasattr(self.cls, 'search_lcsc_api'))
        sig = inspect.signature(self.cls.search_lcsc_api)
        params = list(sig.parameters.keys())
        self.assertIn('lcsc_id', params)

    def test_digikey_get_token_exists(self):
        self.assertTrue(hasattr(self.cls, '_digikey_get_token'))

    def test_search_datasheets_accepts_lcsc_id(self):
        sig = inspect.signature(self.cls._search_datasheets)
        params = list(sig.parameters.keys())
        self.assertIn('lcsc_id', params)

    def test_normalize_part_number_is_module_level(self):
        from symbol_assembler import normalize_part_number
        self.assertTrue(callable(normalize_part_number))

    def test_generate_mpn_search_variants_is_module_level(self):
        from symbol_assembler import generate_mpn_search_variants
        self.assertTrue(callable(generate_mpn_search_variants))


class TestEnvVars(unittest.TestCase):
    """Verify new environment variables are declared."""

    def test_digikey_env_vars(self):
        from symbol_assembler import DIGIKEY_CLIENT_ID, DIGIKEY_CLIENT_SECRET
        self.assertIsInstance(DIGIKEY_CLIENT_ID, str)
        self.assertIsInstance(DIGIKEY_CLIENT_SECRET, str)

    def test_mouser_env_var(self):
        from symbol_assembler import MOUSER_API_KEY
        self.assertIsInstance(MOUSER_API_KEY, str)


class TestLiveAPIs(unittest.TestCase):
    """Live API tests -- only run with --live flag.

    These call real APIs and require credentials in env vars.
    They verify actual connectivity and response parsing.
    """

    @classmethod
    def setUpClass(cls):
        if '--live' not in sys.argv:
            raise unittest.SkipTest("Live API tests skipped (pass --live to enable)")

    def test_lcsc_api_known_part(self):
        """LCSC public API for C523 (100nF MLCC) should return a datasheet URL."""
        from symbol_assembler import SymbolAssembler

        async def run():
            assembler = SymbolAssembler.__new__(SymbolAssembler)
            assembler._http_client = None
            result = await assembler.search_lcsc_api("C523")
            return result

        result = asyncio.run(run())
        self.assertIsNotNone(result, "LCSC API returned None for C523")
        self.assertEqual(result["source"], "lcsc")
        self.assertTrue(result["url"], "LCSC API returned empty URL for C523")
        print(f"  LCSC C523 datasheet: {result['url']}")

    def test_digikey_api_with_creds(self):
        """DigiKey API search for a common part (if creds configured)."""
        from symbol_assembler import DIGIKEY_CLIENT_ID, DIGIKEY_CLIENT_SECRET

        if not DIGIKEY_CLIENT_ID or not DIGIKEY_CLIENT_SECRET:
            self.skipTest("DIGIKEY_CLIENT_ID/SECRET not set")

        from symbol_assembler import SymbolAssembler

        async def run():
            assembler = SymbolAssembler.__new__(SymbolAssembler)
            assembler._http_client = None
            result = await assembler.search_digikey_api("STM32G431CBT6")
            return result

        result = asyncio.run(run())
        if result:
            print(f"  DigiKey STM32G431CBT6 datasheet: {result['url']}")
        else:
            print("  DigiKey: no result (may need production OAuth scope)")

    def test_mouser_api_with_creds(self):
        """Mouser API search for a common part (if creds configured)."""
        from symbol_assembler import MOUSER_API_KEY

        if not MOUSER_API_KEY:
            self.skipTest("MOUSER_API_KEY not set")

        from symbol_assembler import SymbolAssembler

        async def run():
            assembler = SymbolAssembler.__new__(SymbolAssembler)
            assembler._http_client = None
            result = await assembler.search_mouser_api("STM32G431CBT6")
            return result

        result = asyncio.run(run())
        if result:
            print(f"  Mouser STM32G431CBT6 datasheet: {result['url']}")
        else:
            print("  Mouser: no result (check API key)")


if __name__ == '__main__':
    # Remove --live from argv so unittest doesn't choke on it
    live = '--live' in sys.argv
    if live:
        sys.argv.remove('--live')

    # Run tests
    unittest.main(verbosity=2)
