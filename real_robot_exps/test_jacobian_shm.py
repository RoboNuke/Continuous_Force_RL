#!/usr/bin/env python3
"""Check pylibfranka API to understand zero_jacobian return type and data layout."""

import pylibfranka
import inspect

print("=== pylibfranka inspection ===\n")

# List all top-level names
print("Top-level exports:")
for name in sorted(dir(pylibfranka)):
    if not name.startswith('_'):
        obj = getattr(pylibfranka, name)
        print(f"  {name}: {type(obj).__name__}")
print()

# Model class details
print("Model methods/attrs:")
for name in sorted(dir(pylibfranka.Model)):
    if not name.startswith('_'):
        obj = getattr(pylibfranka.Model, name)
        print(f"  {name}: {type(obj).__name__}")
        try:
            doc = obj.__doc__
            if doc:
                # Print first 3 lines of doc
                for line in doc.strip().split('\n')[:3]:
                    print(f"    {line.strip()}")
        except:
            pass
print()

# zero_jacobian details
print("=== zero_jacobian details ===")
zj = pylibfranka.Model.zero_jacobian
print(f"Type: {type(zj)}")
try:
    print(f"Doc:\n{zj.__doc__}")
except:
    pass
print()

# RobotState details
print("=== RobotState details ===")
for name in sorted(dir(pylibfranka.RobotState)):
    if not name.startswith('_'):
        print(f"  {name}")

# Check if there's any way to get return type info
print("\n=== Checking for type hints or signatures ===")
try:
    sig = inspect.signature(pylibfranka.Model.zero_jacobian)
    print(f"zero_jacobian signature: {sig}")
except:
    print("Could not get signature")

try:
    ann = pylibfranka.Model.zero_jacobian.__annotations__
    print(f"zero_jacobian annotations: {ann}")
except:
    print("No annotations")
