#!/usr/bin/env python3
"""
KiCad Path Detection - Cross-platform path resolution for KiCad tools.

This module provides centralized path detection for:
- KiCad Python interpreter
- KiCad site-packages directory
- kicad-cli executable

Supports:
- macOS development environment
- Linux/Docker/K8s deployment
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Tuple, Optional


def get_kicad_python_paths() -> Tuple[str, str]:
    """
    Get KiCad Python executable and site-packages path.

    Returns:
        Tuple of (python_path, site_packages_path)
    """
    if sys.platform == 'darwin':  # macOS
        # Try multiple KiCad Python versions
        mac_paths = [
            ("/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.11/bin/python3",
             "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages"),
            ("/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3",
             "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages"),
            ("/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3",
             "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/lib/python3.9/site-packages"),
        ]
        for python_path, site_packages in mac_paths:
            if Path(python_path).exists():
                return python_path, site_packages
        # Fallback
        return mac_paths[0][0], mac_paths[0][1]

    elif sys.platform == 'linux':  # Linux/K8s/Docker
        # In Docker/K8s, pcbnew is installed in system Python
        python_candidates = [
            "/usr/bin/python3",
            "/usr/local/bin/python3",
        ]
        for python_path in python_candidates:
            if Path(python_path).exists():
                break
        else:
            python_path = "/usr/bin/python3"

        # Site packages varies by distro
        site_packages_candidates = [
            "/usr/lib/python3/dist-packages",  # Debian/Ubuntu
            "/usr/lib64/python3/site-packages",  # Fedora/RHEL
            "/usr/local/lib/python3.11/dist-packages",
            "/usr/local/lib/python3.10/dist-packages",
        ]
        for site_packages in site_packages_candidates:
            if Path(site_packages).exists():
                break
        else:
            site_packages = "/usr/lib/python3/dist-packages"

        return python_path, site_packages

    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")


def get_kicad_cli_path() -> Optional[str]:
    """
    Find kicad-cli executable.

    Returns:
        Path to kicad-cli or None if not found
    """
    candidates = [
        '/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli',  # macOS
        '/usr/bin/kicad-cli',  # Linux system install
        '/usr/local/bin/kicad-cli',  # Local install
        shutil.which('kicad-cli'),  # PATH lookup
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    return None


def ensure_pcbnew_importable():
    """
    Ensure pcbnew can be imported by adding site-packages to sys.path if needed.

    Call this at the top of scripts that need to import pcbnew.
    """
    try:
        import pcbnew
        return True
    except ImportError:
        pass

    # Try adding site-packages
    _, site_packages = get_kicad_python_paths()
    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)

    try:
        import pcbnew
        return True
    except ImportError:
        return False


# Module-level constants for backward compatibility
KICAD_PYTHON, KICAD_SITE_PACKAGES = get_kicad_python_paths()
KICAD_CLI = get_kicad_cli_path()


if __name__ == '__main__':
    print(f"Platform: {sys.platform}")
    print(f"KICAD_PYTHON: {KICAD_PYTHON}")
    print(f"KICAD_SITE_PACKAGES: {KICAD_SITE_PACKAGES}")
    print(f"KICAD_CLI: {KICAD_CLI}")
    print(f"Python exists: {Path(KICAD_PYTHON).exists()}")
    print(f"Site-packages exists: {Path(KICAD_SITE_PACKAGES).exists()}")
    print(f"CLI exists: {Path(KICAD_CLI).exists() if KICAD_CLI else False}")
