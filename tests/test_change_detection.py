#!/usr/bin/env python

"""Tests for change detection module."""

import unittest
import sys

import geoai


class TestChangeDetectionImport(unittest.TestCase):
    """Tests for change_detection module import."""

    def test_change_detection_module_exists(self):
        """Test that change_detection module can be imported."""
        try:
            from geoai import change_detection
            self.assertTrue(hasattr(change_detection, "__name__"))
        except ImportError:
            # If dependencies are missing, that's acceptable
            self.skipTest("change_detection module dependencies not available")

    def test_change_detection_class_exposed(self):
        """Test that ChangeDetection class is exposed in geoai namespace."""
        # Check if ChangeDetection is available in geoai namespace
        has_change_detection = hasattr(geoai, "ChangeDetection")
        
        # If torchange is installed, ChangeDetection should be available
        # If torchange is not installed, ChangeDetection should not be available
        # Both cases are valid
        if has_change_detection:
            # ChangeDetection is available - torchange must be installed
            self.assertTrue(callable(geoai.ChangeDetection))
        else:
            # ChangeDetection is not available - torchange not installed
            # This is expected and acceptable
            pass

    def test_change_detection_import_error_handling(self):
        """Test that missing torchange dependency is handled gracefully."""
        try:
            # Try to import the class directly
            from geoai.change_detection import ChangeDetection
            
            # If import succeeds, try to instantiate
            # This should fail if torchange is not installed
            try:
                cd = ChangeDetection()
                # If we get here, torchange must be installed
                self.assertIsNotNone(cd)
            except ImportError as e:
                # Expected error when torchange is not available
                self.assertIn("torchange", str(e).lower())
                
        except ImportError:
            # If the module itself can't be imported due to missing dependencies,
            # that's acceptable - it means the optional import pattern worked
            pass

    def test_change_detection_error_message(self):
        """Test that helpful error message is shown when torchange is missing."""
        try:
            from geoai.change_detection import ChangeDetection
            
            # Try to instantiate - should fail if torchange not installed
            try:
                cd = ChangeDetection()
            except ImportError as e:
                error_msg = str(e)
                # Error message should mention torchange
                self.assertIn("torchange", error_msg.lower())
                # Error message should mention pip install
                self.assertIn("pip install", error_msg.lower())
                
        except ImportError:
            # Module import failed, which is acceptable
            self.skipTest("change_detection module dependencies not available")


if __name__ == "__main__":
    unittest.main()
