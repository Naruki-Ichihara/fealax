"""
Pytest configuration and fixtures for fealax tests.

This configuration ensures environment validation tests run first
before any other tests to verify the development environment is
properly set up.
"""
import pytest
import sys
from pathlib import Path


def pytest_collection_modifyitems(config, items):
    """Modify test collection to prioritize environment tests.
    
    This function reorders the test collection so that environment
    validation tests run first, ensuring the development environment
    is properly configured before running any functional tests.
    
    Args:
        config: pytest configuration object
        items: List of collected test items
    """
    # Separate environment tests from other tests
    env_tests = []
    other_tests = []
    
    for item in items:
        # Check if test is from test_environment.py
        if "test_environment.py" in str(item.fspath):
            env_tests.append(item)
        else:
            other_tests.append(item)
    
    # Sort environment tests by priority (setup tests first)
    env_test_priority = {
        "test_python_version": 1,
        "test_container_environment": 2,
        "test_display_environment": 3,
        "test_numpy_available": 4,
        "test_jax_available": 5,
        "test_fealax_importable": 99,  # Package tests last in env
    }
    
    def get_env_test_priority(item):
        """Get priority for environment test ordering."""
        test_name = item.name.split("[")[0]  # Remove parametrization
        return env_test_priority.get(test_name, 50)  # Default middle priority
    
    env_tests.sort(key=get_env_test_priority)
    
    # Reorder items: environment tests first, then others
    items[:] = env_tests + other_tests


def pytest_sessionstart(session):
    """Called after session has been created.
    
    Print information about test execution order and environment validation.
    """
    print("\n" + "="*60)
    print("🔬 fealax Test Suite")
    print("="*60)
    print("✓ Environment validation tests will run first")
    print("✓ Warnings are automatically suppressed") 
    print("✓ All tests include comprehensive validation")
    print("="*60)


def pytest_runtest_setup(item):
    """Called for each test before it runs.
    
    Add environment validation markers and ensure proper test isolation.
    """
    # Mark environment tests for potential special handling
    if "test_environment.py" in str(item.fspath):
        if not hasattr(item, "pytestmark"):
            item.pytestmark = []
        # Add environment marker if not already present
        env_marker = pytest.mark.env_validation
        if env_marker not in item.pytestmark:
            item.pytestmark.append(env_marker)


def pytest_runtest_teardown(item, nextitem):
    """Called after each test runs.
    
    Clean up any test artifacts and ensure test isolation.
    """
    # If an environment test failed, we might want to abort early
    # This can be customized based on specific requirements
    pass


@pytest.fixture(scope="session", autouse=True)
def validate_environment():
    """Session-scoped fixture to validate basic environment setup.
    
    This fixture runs once at the start of the test session to perform
    basic environment validation before any tests execute.
    
    Yields:
        dict: Environment validation results
    """
    validation_results = {}
    
    # Basic Python version check
    validation_results["python_version"] = sys.version_info
    
    # Check if we're in the expected container environment
    import os
    validation_results["workspace_path"] = os.getcwd()
    validation_results["has_workspace"] = "/workspace" in os.getcwd()
    
    # Try importing core dependencies
    try:
        import numpy as np
        validation_results["numpy_version"] = np.__version__
        validation_results["numpy_available"] = True
    except ImportError:
        validation_results["numpy_available"] = False
    
    try:
        import jax
        validation_results["jax_available"] = True
        validation_results["jax_devices"] = len(jax.devices())
    except ImportError:
        validation_results["jax_available"] = False
    
    # Yield results for tests to use
    yield validation_results
    
    # Session teardown (if needed)
    pass


@pytest.fixture
def temp_workspace(tmp_path):
    """Provide a temporary workspace for tests that need file I/O.
    
    Args:
        tmp_path: pytest temporary path fixture
        
    Returns:
        Path: Temporary directory path
    """
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    return workspace


# Define custom markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "env_validation: mark test as environment validation"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU access"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )