#!/usr/bin/env python3
"""
Minimal tests that work without external dependencies
Can be run directly: python tests/test_minimal.py
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test counter
tests_passed = 0
tests_failed = 0

def simple_test(description):
    """Simple test decorator for standalone execution"""
    def decorator(func):
        def wrapper():
            global tests_passed, tests_failed
            print(f"Testing: {description}... ", end="")
            try:
                func()
                print("✅ PASSED")
                tests_passed += 1
            except Exception as e:
                print(f"❌ FAILED: {e}")
                tests_failed += 1
        return wrapper
    return decorator


@simple_test("Performance profiles configuration")
def test_performance_profiles():
    """Test that performance profiles are correctly defined"""
    from app import PERFORMANCE_PROFILES
    
    # Check all profiles exist
    assert "whisper-1" in PERFORMANCE_PROFILES
    assert "whisper-1-fast" in PERFORMANCE_PROFILES
    assert "whisper-1-quality" in PERFORMANCE_PROFILES
    
    # Check fast profile
    fast = PERFORMANCE_PROFILES["whisper-1-fast"]
    assert fast["beam_size"] == 1
    assert fast["best_of"] == 1
    
    # Check quality profile
    quality = PERFORMANCE_PROFILES["whisper-1-quality"]
    assert quality["beam_size"] == 10
    assert quality["best_of"] == 10


@simple_test("Timestamp formatting")
def test_timestamp_formatting():
    """Test timestamp formatting functions"""
    from app import format_timestamp
    
    # Test SRT format
    assert format_timestamp(0.0, "srt") == "00:00:00,000"
    assert format_timestamp(90.5, "srt") == "00:01:30,500"
    
    # Test VTT format
    assert format_timestamp(0.0, "vtt") == "00:00:00.000"
    assert format_timestamp(90.5, "vtt") == "00:01:30.500"


@simple_test("Audio conversion imports")
def test_audio_conversion_imports():
    """Test that audio conversion dependencies can be imported"""
    try:
        import numpy as np
        import soundfile as sf
        from app import convert_audio
        assert True
    except ImportError as e:
        # This is OK for minimal tests
        print(f"(Skipped - dependencies not installed: {e})")
        raise


@simple_test("Environment variables")
def test_environment_defaults():
    """Test default environment variable values"""
    # These should work regardless of environment
    device = os.getenv("DEVICE", "cpu")
    assert device in ["cpu", "cuda"]
    
    model_size = os.getenv("MODEL_SIZE", "large-v3")
    assert model_size == "large-v3"
    
    # Compute type should match device
    compute_type = os.getenv("COMPUTE_TYPE", "int8" if device == "cpu" else "float16")
    assert compute_type in ["int8", "float16", "float32"]


@simple_test("API endpoints structure")
def test_api_structure():
    """Test that API endpoint handlers are defined"""
    import app
    
    # Check main endpoints exist
    assert hasattr(app, 'health')
    assert hasattr(app, 'list_models')
    assert hasattr(app, 'transcribe')
    assert hasattr(app, 'translate')
    
    # Check model definitions
    assert hasattr(app, 'TranscriptionResponse')
    assert hasattr(app, 'TranslationResponse')
    assert hasattr(app, 'ModelInfo')


@simple_test("Model configuration")
def test_model_configuration():
    """Test model configuration constants"""
    import app
    
    # Check constants
    assert hasattr(app, 'MODEL_SIZE')
    assert hasattr(app, 'DEVICE')
    assert hasattr(app, 'COMPUTE_TYPE')
    assert hasattr(app, 'ENABLE_DIARIZATION')
    
    # Check diarization logic
    if app.DEVICE == "cpu":
        assert app.ENABLE_DIARIZATION == False  # Should be disabled on CPU
    

@simple_test("Security configuration")
def test_security_configuration():
    """Test API security setup"""
    import app
    
    # Check API keys configuration
    assert hasattr(app, 'API_KEYS')
    assert isinstance(app.API_KEYS, list)
    
    # Check security dependency
    assert hasattr(app, 'verify_api_key')
    assert hasattr(app, 'security')


def run_integration_test():
    """Simple integration test using curl"""
    print("\n" + "="*60)
    print("Running integration test (requires service to be running)")
    print("="*60)
    
    import subprocess
    
    try:
        # Check if service is running
        result = subprocess.run(
            ["curl", "-s", "http://localhost:8000/"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✅ Service is responding")
            # Try to parse response
            try:
                data = json.loads(result.stdout)
                print(f"  - Status: {data.get('status', 'unknown')}")
                print(f"  - Model: {data.get('model', 'unknown')}")
                print(f"  - Device: {data.get('device', 'unknown')}")
                print(f"  - Diarization: {data.get('diarization_enabled', False)}")
            except json.JSONDecodeError:
                print("  ⚠️  Could not parse JSON response")
        else:
            print("❌ Service is not running or not responding")
            print("   Run: docker-compose -f docker-compose.cpu.yml up -d")
    except subprocess.TimeoutExpired:
        print("❌ Service timeout - not running")
    except FileNotFoundError:
        print("⚠️  curl not found - skipping integration test")


# Main execution
if __name__ == "__main__":
    print("🧪 Running minimal tests for Faster Whisper v2")
    print("=" * 60)
    
    # Run all tests
    test_functions = []
    for name, obj in list(globals().items()):
        if name.startswith('test_') and callable(obj):
            test_functions.append((name, obj))
    
    # Run each test function
    for name, func in test_functions:
        func()
    
    # Summary
    print("\n" + "="*60)
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    
    # Run integration test if requested
    if "--integration" in sys.argv:
        run_integration_test()
    
    # Exit code
    sys.exit(0 if tests_failed == 0 else 1)
