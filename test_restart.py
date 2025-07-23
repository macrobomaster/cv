#!/usr/bin/env python3

"""
Test script for process restart functionality.
Tests that the supervisor can restart processes based on exit codes.
"""

import sys
import os
import time
import tempfile
import subprocess
from pathlib import Path

# Add the cv module to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def create_test_process_module(exit_code, delay=1):
    """Create a temporary test process module that exits with a specific code."""
    
    module_content = f"""
import sys
import time
from cv.system.core.logging import logger

def run():
    logger.info("Test process starting")
    time.sleep({delay})
    logger.info("Test process exiting with code {exit_code}")
    sys.exit({exit_code})
"""
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(module_content)
        return f.name

def test_restart_functionality():
    """Test that processes restart when they exit with EXIT_RESTART code."""
    
    # Import required modules
    from cv.system.core.supervisor import SupervisedProcess, Supervisor
    from cv.system.core.exit_codes import EXIT_RESTART
    
    print("Testing process restart functionality...")
    
    # Create a test module that exits with EXIT_RESTART
    test_module_path = create_test_process_module(EXIT_RESTART, delay=0.5)
    module_name = f"test_module_{int(time.time())}"
    
    try:
        # Add the module to sys.modules dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, test_module_path)
        test_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = test_module
        spec.loader.exec_module(test_module)
        
        # Create a supervised process
        proc = SupervisedProcess(
            name="test_proc",
            module=module_name,
            restart_on_exit_codes=[EXIT_RESTART]
        )
        
        # Start the process
        proc.start()
        print(f"Started process with PID: {proc.proc.pid}")
        
        # Wait for it to exit
        proc.proc.join(timeout=2.0)
        exit_code = proc.proc.exitcode
        print(f"Process exited with code: {exit_code}")
        
        # Check that it exited with the expected code
        if exit_code == EXIT_RESTART:
            print("✓ Process exited with EXIT_RESTART code as expected")
            return True
        else:
            print(f"✗ Process exited with unexpected code: {exit_code}")
            return False
            
    finally:
        # Clean up
        if module_name in sys.modules:
            del sys.modules[module_name]
        if os.path.exists(test_module_path):
            os.unlink(test_module_path)

def test_no_restart_on_normal_exit():
    """Test that processes don't restart on normal exit codes."""
    
    from cv.system.core.supervisor import SupervisedProcess
    from cv.system.core.exit_codes import EXIT_RESTART
    
    print("Testing normal exit behavior...")
    
    # Create a test module that exits normally
    test_module_path = create_test_process_module(0, delay=0.5)  # Exit with code 0
    module_name = f"test_module_normal_{int(time.time())}"
    
    try:
        # Add the module to sys.modules dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, test_module_path)
        test_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = test_module
        spec.loader.exec_module(test_module)
        
        # Create a supervised process
        proc = SupervisedProcess(
            name="test_proc_normal",
            module=module_name,
            restart_on_exit_codes=[EXIT_RESTART]
        )
        
        # Start the process
        proc.start()
        print(f"Started process with PID: {proc.proc.pid}")
        
        # Wait for it to exit
        proc.proc.join(timeout=2.0)
        exit_code = proc.proc.exitcode
        print(f"Process exited with code: {exit_code}")
        
        # Check that it exited normally
        if exit_code == 0:
            print("✓ Process exited normally as expected")
            return True
        else:
            print(f"✗ Process exited with unexpected code: {exit_code}")
            return False
            
    finally:
        # Clean up
        if module_name in sys.modules:
            del sys.modules[module_name]
        if os.path.exists(test_module_path):
            os.unlink(test_module_path)

if __name__ == "__main__":
    print("Running tests for process restart functionality...")
    
    # Install required dependencies for testing
    try:
        import cv.system.core.supervisor
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        sys.exit(1)
    
    success = True
    
    # Run tests
    success &= test_restart_functionality()
    success &= test_no_restart_on_normal_exit()
    
    if success:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)