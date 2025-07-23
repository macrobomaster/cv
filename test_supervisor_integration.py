#!/usr/bin/env python3

"""
Integration test for supervisor restart functionality.
Tests the full supervisor loop with restart logic.
"""

import sys
import os
import time
import tempfile
import threading
from pathlib import Path

# Add the cv module to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def create_counting_module(exit_code, max_runs=2):
    """Create a module that counts runs and exits with specified code."""
    
    module_content = f"""
import sys
import os
import time
from cv.system.core.logging import logger

counter_file = "/tmp/test_counter_{{int(time.time())}}.txt"

def run():
    # Read current count
    count = 0
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as f:
            count = int(f.read().strip())
    
    count += 1
    logger.info(f"Test process run #{{count}}")
    
    # Write new count
    with open(counter_file, 'w') as f:
        f.write(str(count))
    
    time.sleep(0.5)
    
    if count >= {max_runs}:
        logger.info(f"Reached max runs ({max_runs}), exiting with code 0")
        sys.exit(0)
    else:
        logger.info(f"Run #{{count}}, exiting with code {exit_code} to trigger restart")
        sys.exit({exit_code})
"""
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(module_content)
        return f.name

def test_supervisor_restart_loop():
    """Test that supervisor properly restarts processes in the main loop."""
    
    from cv.system.core.supervisor import SupervisedProcess, Supervisor
    from cv.system.core.exit_codes import EXIT_RESTART
    
    print("Testing supervisor restart loop...")
    
    # Create a test module that runs twice before exiting normally
    test_module_path = create_counting_module(EXIT_RESTART, max_runs=3)
    module_name = f"supervisor_test_module_{int(time.time())}"
    
    try:
        # Add the module to sys.modules dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, test_module_path)
        test_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = test_module
        spec.loader.exec_module(test_module)
        
        # Create a supervised process
        proc = SupervisedProcess(
            name="restart_test_proc",
            module=module_name,
            restart_on_exit_codes=[EXIT_RESTART]
        )
        
        # Create supervisor
        supervisor = Supervisor([proc])
        
        # Run supervisor in a separate thread with timeout
        def run_supervisor():
            supervisor.run()
        
        supervisor_thread = threading.Thread(target=run_supervisor, daemon=True)
        supervisor_thread.start()
        
        # Let it run for a few seconds to allow restarts
        time.sleep(4)
        
        # Signal shutdown
        from cv.system.core.keyvalue import kv_put
        kv_put("global", "do_shutdown", True)
        
        # Wait for supervisor to shutdown
        supervisor_thread.join(timeout=2)
        
        # Check if it restarted at least once
        if proc.last_exit_code == EXIT_RESTART:
            print("✓ Supervisor detected EXIT_RESTART code")
            return True
        else:
            print(f"✗ Expected EXIT_RESTART, got: {proc.last_exit_code}")
            return False
            
    finally:
        # Clean up
        if module_name in sys.modules:
            del sys.modules[module_name]
        if os.path.exists(test_module_path):
            os.unlink(test_module_path)

if __name__ == "__main__":
    print("Running integration test for supervisor restart functionality...")
    
    success = test_supervisor_restart_loop()
    
    if success:
        print("\n✓ Integration test passed!")
        sys.exit(0)
    else:
        print("\n✗ Integration test failed!")
        sys.exit(1)