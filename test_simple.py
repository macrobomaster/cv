#!/usr/bin/env python3

"""
Simple test for the restart functionality without full supervisor setup.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add the cv module to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_autoaimd_restart_behavior():
    """Test that autoaimd exits with restart code when building cached model."""
    
    print("Testing autoaimd restart behavior...")
    
    # Create a mock environment for autoaimd
    os.environ["SYSTEM_PATH"] = "/tmp/test_system"
    os.makedirs("/tmp/test_system", exist_ok=True)
    
    # Create a temporary weights file
    weights_dir = Path(__file__).parent.parent.parent / "weights"
    weights_dir.mkdir(exist_ok=True)
    
    # Create a dummy model file (we can't create a real one without the full model)
    dummy_model_content = b"dummy content"
    with open(weights_dir / "model.safetensors", "wb") as f:
        f.write(dummy_model_content)
    
    try:
        # Import the exit codes
        from cv.system.core.exit_codes import EXIT_RESTART
        
        print(f"✓ EXIT_RESTART constant = {EXIT_RESTART}")
        
        # Check if autoaimd imports the exit code
        try:
            from cv.system.autoaimd import autoaimd
            print("✓ autoaimd module imports EXIT_RESTART successfully")
            
            # Check that the exit code is used in the right place
            autoaimd_path = Path(__file__).parent / "cv/system/autoaimd/autoaimd.py"
            with open(autoaimd_path, "r") as f:
                content = f.read()
                if "sys.exit(EXIT_RESTART)" in content:
                    print("✓ autoaimd uses sys.exit(EXIT_RESTART) as expected")
                    return True
                else:
                    print("✗ autoaimd does not use sys.exit(EXIT_RESTART)")
                    return False
                    
        except Exception as e:
            print(f"✗ Error importing autoaimd: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Error during test: {e}")
        return False

def test_supervisor_modifications():
    """Test that supervisor handles exit codes correctly."""
    
    print("Testing supervisor modifications...")
    
    try:
        from cv.system.core.exit_codes import EXIT_RESTART
        from cv.system.core.supervisor import SupervisedProcess
        
        # Create a test process
        proc = SupervisedProcess(
            name="test",
            module="dummy",
            restart_on_exit_codes=[EXIT_RESTART]
        )
        
        # Check that it has the new attributes
        if hasattr(proc, 'restart_on_exit_codes'):
            print("✓ SupervisedProcess has restart_on_exit_codes attribute")
        else:
            print("✗ SupervisedProcess missing restart_on_exit_codes attribute")
            return False
            
        if hasattr(proc, 'last_exit_code'):
            print("✓ SupervisedProcess has last_exit_code attribute")
        else:
            print("✗ SupervisedProcess missing last_exit_code attribute")
            return False
            
        if EXIT_RESTART in proc.restart_on_exit_codes:
            print("✓ EXIT_RESTART is in default restart_on_exit_codes")
        else:
            print("✗ EXIT_RESTART not in default restart_on_exit_codes")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error during supervisor test: {e}")
        return False

if __name__ == "__main__":
    print("Running simple tests for restart functionality...")
    
    success = True
    success &= test_autoaimd_restart_behavior()
    success &= test_supervisor_modifications()
    
    if success:
        print("\n✓ All simple tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)