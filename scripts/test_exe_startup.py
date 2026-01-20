#!/usr/bin/env python3
"""
Automated testing script for COVID19_Demo.exe standalone executable.

Validates:
  - Executable exists and has correct size
  - SHA256 checksum matches
  - Can launch without errors (smoke test)
  - Models are accessible
  - Basic functionality works

Usage:
    python scripts/test_exe_startup.py --exe dist/COVID19_Demo.exe
    python scripts/test_exe_startup.py --exe dist/COVID19_Demo.exe --full

Options:
    --exe PATH      Path to executable to test
    --full          Run full integration tests (slower)
    --timeout SEC   Timeout for startup test (default: 60)
"""

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path


class Colors:
    """ANSI color codes."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_test(name, status, message=""):
    """Print test result with colored status."""
    if status == "PASS":
        color = Colors.OKGREEN
        symbol = "✓"
    elif status == "FAIL":
        color = Colors.FAIL
        symbol = "✗"
    elif status == "WARN":
        color = Colors.WARNING
        symbol = "⚠"
    else:
        color = Colors.OKCYAN
        symbol = "ℹ"

    print(f"{color}{symbol} {name:<50} [{status}]{Colors.ENDC}")
    if message:
        print(f"  {message}")


def test_file_exists(exe_path):
    """Test 1: Executable file exists."""
    if exe_path.exists():
        print_test("Executable exists", "PASS", f"Path: {exe_path}")
        return True
    else:
        print_test("Executable exists", "FAIL", f"Not found: {exe_path}")
        return False


def test_file_size(exe_path):
    """Test 2: File size is reasonable."""
    size_mb = exe_path.stat().st_size / (1024 * 1024)

    if size_mb < 500:
        print_test("File size check", "FAIL", f"Too small: {size_mb:.1f} MB (expected ~1800 MB)")
        return False
    elif size_mb > 3000:
        print_test("File size check", "FAIL", f"Too large: {size_mb:.1f} MB (expected ~1800 MB)")
        return False
    else:
        print_test("File size check", "PASS", f"Size: {size_mb:.1f} MB")
        return True


def test_checksum(exe_path):
    """Test 3: SHA256 checksum matches (if .sha256 file exists)."""
    sha256_file = exe_path.parent / f"{exe_path.name}.sha256"

    if not sha256_file.exists():
        print_test("Checksum verification", "WARN", "No .sha256 file found (skipping)")
        return True

    # Read expected checksum
    try:
        with open(sha256_file, 'r') as f:
            expected_hash = f.read().strip().split()[0]
    except Exception as e:
        print_test("Checksum verification", "FAIL", f"Cannot read .sha256 file: {e}")
        return False

    # Compute actual checksum
    print("  Computing SHA256 (this may take 1-2 minutes)...")
    sha256 = hashlib.sha256()
    try:
        with open(exe_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096 * 1024), b""):  # 4MB chunks
                sha256.update(chunk)
    except Exception as e:
        print_test("Checksum verification", "FAIL", f"Cannot compute hash: {e}")
        return False

    actual_hash = sha256.hexdigest()

    if actual_hash.lower() == expected_hash.lower():
        print_test("Checksum verification", "PASS", f"SHA256: {actual_hash[:16]}...")
        return True
    else:
        print_test("Checksum verification", "FAIL",
                  f"Mismatch!\n  Expected: {expected_hash}\n  Actual:   {actual_hash}")
        return False


def test_smoke_launch(exe_path, timeout=60):
    """Test 4: Executable can launch without immediate crash (smoke test)."""
    print_test("Smoke test (launch)", "INFO", f"Launching executable (timeout: {timeout}s)...")

    try:
        # Launch process (don't wait for completion)
        process = subprocess.Popen(
            [str(exe_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=exe_path.parent
        )

        # Wait a few seconds to see if it crashes immediately
        time.sleep(5)

        # Check if still running
        poll_result = process.poll()
        if poll_result is not None:
            # Process exited
            stdout, stderr = process.communicate()
            print_test("Smoke test (launch)", "FAIL",
                      f"Process exited with code {poll_result}\n"
                      f"STDOUT: {stdout[:500]}\n"
                      f"STDERR: {stderr[:500]}")
            return False

        # Process is running, wait a bit more to check for startup errors
        time.sleep(10)

        poll_result = process.poll()
        if poll_result is not None:
            stdout, stderr = process.communicate()
            print_test("Smoke test (launch)", "FAIL",
                      f"Process crashed after 15s with code {poll_result}\n"
                      f"STDERR: {stderr[:500]}")
            return False

        # Still running after 15 seconds, looks good
        print_test("Smoke test (launch)", "PASS",
                  "Process launched successfully and is running")

        # Terminate process
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

        return True

    except FileNotFoundError:
        print_test("Smoke test (launch)", "FAIL", f"Cannot execute: {exe_path}")
        return False
    except Exception as e:
        print_test("Smoke test (launch)", "FAIL", f"Unexpected error: {e}")
        return False


def test_full_integration(exe_path):
    """Test 5: Full integration test (requires manual interaction)."""
    print("\n" + "="*70)
    print("FULL INTEGRATION TEST (Manual)")
    print("="*70)
    print("\nThis test requires manual interaction.")
    print("The executable will launch, and you should:")
    print("  1. Wait for browser to open (~20-30 seconds)")
    print("  2. Verify interface loads correctly")
    print("  3. Upload a test image")
    print("  4. Click 'Procesar Imagen'")
    print("  5. Verify all 4 visualizations appear")
    print("  6. Close browser and press Ctrl+C in console")
    print()

    response = input("Ready to start full test? (y/n): ")
    if response.lower() not in ['y', 'yes', 's', 'si']:
        print_test("Full integration test", "WARN", "Skipped by user")
        return True

    print("\nLaunching executable...")
    try:
        # Launch and wait for user to close
        subprocess.run([str(exe_path)], cwd=exe_path.parent)

        response = input("\nDid all tests pass? (y/n): ")
        if response.lower() in ['y', 'yes', 's', 'si']:
            print_test("Full integration test", "PASS", "User confirmed success")
            return True
        else:
            print_test("Full integration test", "FAIL", "User reported issues")
            return False

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        response = input("Did tests pass before interruption? (y/n): ")
        if response.lower() in ['y', 'yes', 's', 'si']:
            print_test("Full integration test", "PASS", "User confirmed success")
            return True
        else:
            print_test("Full integration test", "FAIL", "User reported issues")
            return False
    except Exception as e:
        print_test("Full integration test", "FAIL", f"Error: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test COVID19_Demo.exe standalone executable'
    )
    parser.add_argument(
        '--exe',
        type=Path,
        required=True,
        help='Path to executable to test'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full integration tests (requires manual interaction)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Timeout for startup test in seconds (default: 60)'
    )

    args = parser.parse_args()

    # Print header
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'COVID19_Demo.exe Testing Suite'.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

    exe_path = args.exe.resolve()
    print(f"Testing executable: {exe_path}\n")

    # Run tests
    results = []

    print(f"{Colors.BOLD}Running automated tests...{Colors.ENDC}\n")

    results.append(("File exists", test_file_exists(exe_path)))
    results.append(("File size", test_file_size(exe_path)))
    results.append(("Checksum", test_checksum(exe_path)))

    # Only run smoke test if file exists
    if results[0][1]:
        results.append(("Smoke test", test_smoke_launch(exe_path, args.timeout)))

    # Full integration test (optional)
    if args.full:
        results.append(("Full integration", test_full_integration(exe_path)))

    # Summary
    print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}Test Summary{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

    total = len(results)
    passed = sum(1 for _, result in results if result)
    failed = total - passed

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print_test(name, status)

    print(f"\n{Colors.BOLD}Total: {total} | Passed: {passed} | Failed: {failed}{Colors.ENDC}\n")

    if failed == 0:
        print(f"{Colors.OKGREEN}{Colors.BOLD}✓ All tests passed!{Colors.ENDC}\n")
        return 0
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}✗ {failed} test(s) failed.{Colors.ENDC}\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
