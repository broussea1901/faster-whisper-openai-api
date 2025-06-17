"""
Unit tests for Faster Whisper v2 API
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImports:
    """Test basic imports"""
    
    def test_app_imports(self):
        """Test that app module can be imported"""
        import app
        assert app is not None
    
    def test_required_attributes(self):
        """Test that required attributes exist"""
        import app
        
        # Check key attributes
        assert hasattr(app, 'PERFORMANCE_PROFILES')
        assert hasattr(app, 'app')  # FastAPI app
        assert hasattr(app, 'whisper_model')


class TestPerformanceProfiles:
    """Test performance profile configurations"""
    
    def test_all_profiles_exist(self):
        """Test that all expected profiles exist"""
        from app import PERFORMANCE_PROFILES
        
        assert len(PERFORMANCE_PROFILES) == 3
        assert "whisper-1" in PERFORMANCE_PROFILES
        assert "whisper-1-fast" in PERFORMANCE_PROFILES
        assert "whisper-1-quality" in PERFORMANCE_PROFILES
    
    def test_profile_configurations(self):
        """Test profile settings are correct"""
        from app import PERFORMANCE_PROFILES
        
        # Fast profile
        fast = PERFORMANCE_PROFILES["whisper-1-fast"]
        assert fast["beam_size"] == 1
        assert fast["best_of"] == 1
        assert fast["patience"] == 0.5
        
        # Balanced profile
        balanced = PERFORMANCE_PROFILES["whisper-1"]
        assert balanced["beam_size"] == 5
        assert balanced["best_of"] == 5
        assert balanced["patience"] == 1.0
        
        # Quality profile
        quality = PERFORMANCE_PROFILES["whisper-1-quality"]
        assert quality["beam_size"] == 10
        assert quality["best_of"] == 10
        assert quality["patience"] == 2.0


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_format_timestamp_srt(self):
        """Test SRT timestamp formatting"""
        from app import format_timestamp
        
        assert format_timestamp(0.0, "srt") == "00:00:00,000"
        assert format_timestamp(1.5, "srt") == "00:00:01,500"
        assert format_timestamp(61.123, "srt") == "00:01:01,123"
        assert format_timestamp(3661.456, "srt") == "01:01:01,456"
    
    def test_format_timestamp_vtt(self):
        """Test VTT timestamp formatting"""
        from app import format_timestamp
        
        assert format_timestamp(0.0, "vtt") == "00:00:00.000"
        assert format_timestamp(1.5, "vtt") == "00:00:01.500"
        assert format_timestamp(61.123, "vtt") == "00:01:01.123"
        assert format_timestamp(3661.456, "vtt") == "01:01:01.456"


class TestConfiguration:
    """Test configuration and environment variables"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        import app
        
        # Check configuration constants exist
        assert hasattr(app, 'MODEL_SIZE')
        assert hasattr(app, 'DEVICE')
        assert hasattr(app, 'COMPUTE_TYPE')
        assert hasattr(app, 'ENABLE_DIARIZATION')
        
        # Check values are sensible
        assert app.DEVICE in ['cpu', 'cuda']
        assert app.MODEL_SIZE in ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
        
    def test_api_keys_configuration(self):
        """Test API keys configuration"""
        import app
        
        assert hasattr(app, 'API_KEYS')
        assert isinstance(app.API_KEYS, list)
