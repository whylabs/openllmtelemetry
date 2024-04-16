from typing import Optional


def package_version(package: Optional[str] = __package__) -> str:
    """Calculate version number based on pyproject.toml"""
    if not package:
        raise ValueError("No package specified when searching for package version")
    try:
        from importlib import metadata

        version = metadata.version(package)
    except Exception:
        version = f"{package} is not installed."

    return version


__version__ = package_version()
