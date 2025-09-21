"""
Pytest configuration for integration tests to handle module mocking cleanup.
"""
import sys
import pytest

# Store original modules that get mocked by integration tests
_original_modules = {}

def pytest_sessionstart(session):
    """Store original sys.modules state before tests start"""
    global _original_modules
    _original_modules = {}
    for module_name in list(sys.modules.keys()):
        if any(module_name.startswith(prefix) for prefix in ['skrl', 'memories', 'envs', 'models', 'agents']):
            _original_modules[module_name] = sys.modules.get(module_name)

def pytest_sessionfinish(session, exitstatus):
    """Restore sys.modules to original state after all integration tests finish"""
    global _original_modules

    # Remove mocked modules that weren't there originally
    modules_to_remove = []
    for module_name in list(sys.modules.keys()):
        if any(module_name.startswith(prefix) for prefix in ['skrl', 'memories', 'envs', 'models', 'agents']):
            if module_name not in _original_modules:
                modules_to_remove.append(module_name)

    for module_name in modules_to_remove:
        del sys.modules[module_name]

    # Restore original modules
    for module_name, original_module in _original_modules.items():
        if original_module is not None:
            sys.modules[module_name] = original_module
        elif module_name in sys.modules:
            del sys.modules[module_name]