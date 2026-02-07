#!/usr/bin/env python3
"""
Setup validation script for Visual Verification Agent

Checks all dependencies and provides setup instructions if missing.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_python_module(module_name: str, package_name: str = None) -> bool:
    """Check if Python module is available."""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"✓ Python module '{module_name}' is available")
        return True
    except ImportError:
        print(f"✗ Python module '{module_name}' is missing")
        print(f"  Install: pip install {package_name}")
        return False


def check_command(command: str, install_hint: str = None) -> bool:
    """Check if system command is available."""
    try:
        result = subprocess.run(
            ["which", command],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            path = result.stdout.strip()
            print(f"✓ Command '{command}' is available at {path}")
            return True
        else:
            print(f"✗ Command '{command}' not found")
            if install_hint:
                print(f"  Install: {install_hint}")
            return False
    except Exception as e:
        print(f"✗ Error checking '{command}': {e}")
        return False


def check_api_key(key_name: str) -> bool:
    """Check if API key is set."""
    value = os.getenv(key_name)
    if value:
        masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
        print(f"✓ Environment variable '{key_name}' is set: {masked}")
        return True
    else:
        print(f"✗ Environment variable '{key_name}' is not set")
        print(f"  Set: export {key_name}='your_api_key'")
        return False


def check_file_exists(file_path: Path) -> bool:
    """Check if required file exists."""
    if file_path.exists():
        print(f"✓ File exists: {file_path}")
        return True
    else:
        print(f"✗ File missing: {file_path}")
        return False


def check_kicad_version() -> bool:
    """Check KiCad version if available."""
    try:
        result = subprocess.run(
            ["kicad-cli", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✓ KiCad CLI version: {version}")

            # Check if version is 8.x
            if "Version: 8." in version or "8.0" in version:
                print(f"  ✓ Version 8.x detected (compatible)")
                return True
            else:
                print(f"  ⚠ Version may not be 8.x (compatibility unknown)")
                return True
        else:
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ KiCad CLI timed out")
        return False
    except Exception as e:
        return False


def main():
    """Run all validation checks."""
    print("="*80)
    print("VISUAL VERIFICATION AGENT - SETUP VALIDATION")
    print("="*80)

    results = {}

    # Python dependencies
    print("\n" + "-"*80)
    print("PYTHON DEPENDENCIES")
    print("-"*80)
    results['httpx'] = check_python_module('httpx')
    results['yaml'] = check_python_module('yaml', 'PyYAML')
    results['pathlib'] = check_python_module('pathlib')  # stdlib
    results['asyncio'] = check_python_module('asyncio')  # stdlib

    # System commands
    print("\n" + "-"*80)
    print("SYSTEM COMMANDS")
    print("-"*80)
    results['kicad-cli'] = check_command(
        'kicad-cli',
        'brew install kicad (macOS) or see https://www.kicad.org/download/'
    )

    if results['kicad-cli']:
        results['kicad-version'] = check_kicad_version()

    results['convert'] = check_command(
        'convert',
        'brew install imagemagick (optional but recommended)'
    )

    # API keys
    print("\n" + "-"*80)
    print("API CREDENTIALS")
    print("-"*80)
    results['api_key'] = check_api_key('OPENROUTER_API_KEY')

    # Required files
    print("\n" + "-"*80)
    print("REQUIRED FILES")
    print("-"*80)
    script_dir = Path(__file__).parent
    results['visual_verifier'] = check_file_exists(script_dir / "visual_verifier.py")
    results['quality_rubric'] = check_file_exists(script_dir / "quality_rubric.yaml")
    results['__init__'] = check_file_exists(script_dir / "__init__.py")

    # Test schematic (optional)
    print("\n" + "-"*80)
    print("TEST DATA (optional)")
    print("-"*80)
    test_schematic = script_dir / "../../output/test_1.kicad_sch"
    test_available = check_file_exists(test_schematic)

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    critical_checks = ['httpx', 'yaml', 'api_key', 'visual_verifier', 'quality_rubric']
    recommended_checks = ['kicad-cli', 'convert']
    optional_checks = ['__init__']

    critical_passed = all(results.get(check, False) for check in critical_checks)
    recommended_passed = all(results.get(check, False) for check in recommended_checks)

    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    print(f"\nChecks passed: {passed_count}/{total_count}")

    if critical_passed:
        print("✓ All CRITICAL checks passed")
        print("  → Visual verifier can run with Python-only features")
    else:
        print("✗ Some CRITICAL checks failed")
        print("  → Visual verifier CANNOT run until issues are resolved")

    if not recommended_passed:
        print("⚠ Some RECOMMENDED checks failed")
        print("  → Install KiCad CLI for full image generation support")

    # Next steps
    print("\n" + "-"*80)
    print("NEXT STEPS")
    print("-"*80)

    if not critical_passed:
        print("\n1. Install missing Python dependencies:")
        print("   cd /path/to/python-scripts")
        print("   pip install -r requirements.txt")
        print("\n2. Set API key:")
        print("   export OPENROUTER_API_KEY='your_key'")
        print("   # Add to ~/.zshrc or ~/.bashrc to persist")

    if not results.get('kicad-cli', False):
        print("\n3. Install KiCad 8.x:")
        print("   macOS: brew install kicad")
        print("   Linux: https://www.kicad.org/download/")
        print("   Windows: https://www.kicad.org/download/windows/")

    if not results.get('convert', False):
        print("\n4. (Optional) Install ImageMagick for PNG conversion:")
        print("   macOS: brew install imagemagick")
        print("   Linux: apt-get install imagemagick")

    if critical_passed and recommended_passed:
        print("\n✓ Setup is complete! You can now run:")
        print("\n  # Basic test")
        print("  python visual_verifier.py /path/to/schematic.kicad_sch")
        print("\n  # Run examples")
        print("  python example_usage.py")
        print("\n  # Run tests")
        print("  pytest test_visual_verifier.py -v")

    # Exit code
    if critical_passed:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
