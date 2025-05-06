"""
Version information for Decision Jungles.

This module contains functions for accessing and managing the version 
information for the Decision Jungles package. It follows semantic versioning
(https://semver.org/).
"""

__version__ = "0.1.0"


def get_version():
    """
    Return the version of the Decision Jungles package.
    
    Returns
    -------
    str
        The package version string.
    """
    return __version__


def parse_version(version_str):
    """
    Parse a version string into its major, minor, and patch components.
    
    Parameters
    ----------
    version_str : str
        The version string to parse.
        
    Returns
    -------
    tuple
        A tuple of (major, minor, patch) version numbers.
    """
    try:
        # Handle pre-release versions (e.g., "1.2.3-alpha.1")
        version_str = version_str.split("-")[0]
        parts = version_str.split(".")
        
        # Ensure we have exactly three parts (major, minor, patch)
        if len(parts) != 3:
            raise ValueError(f"Invalid version string format: {version_str}")
        
        # Convert to integers
        major, minor, patch = map(int, parts)
        return major, minor, patch
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse version string '{version_str}': {str(e)}")


def check_version_compatibility(required_version, current_version=None):
    """
    Check if the current version is compatible with the required version.
    
    Parameters
    ----------
    required_version : str
        The minimum required version string.
    current_version : str, optional
        The current version string. If None, uses the package version.
        
    Returns
    -------
    bool
        True if the current version is compatible with the required version.
    """
    if current_version is None:
        current_version = __version__
        
    req_major, req_minor, req_patch = parse_version(required_version)
    curr_major, curr_minor, curr_patch = parse_version(current_version)
    
    # Major version must match exactly
    if curr_major != req_major:
        return False
    
    # Current minor version must be >= required minor version
    if curr_minor < req_minor:
        return False
    
    # If minor versions match, current patch must be >= required patch
    if curr_minor == req_minor and curr_patch < req_patch:
        return False
    
    return True