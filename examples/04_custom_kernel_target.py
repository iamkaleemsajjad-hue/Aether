"""Example: Register and use a custom hardware target / backend plan."""

from __future__ import annotations

from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
from aether.compiler.stage3_targeting.target_registry import TargetRegistry


def main() -> None:
    registry = TargetRegistry()
    print("Supported targets:")
    for target_id in registry.supported_targets:
        profile = registry.get_profile(target_id)
        print(f"  {target_id}: {profile.name}")

    # Show current hardware
    current = HardwareProfile.auto()
    print(f"\nCurrent hardware target: {current.target_id}")
    print(f"Recommended backend: {current.recommended_backend}")


if __name__ == "__main__":
    main()
