"""Product and process knowledge for high-level production planning."""

from __future__ import annotations


PRODUCT_SPECS = {
    "car": {
        "line": "A",
        "base_part": {"part": "car_base", "source_line": "A"},
        "assemblies": [
            {
                "part": "car_frame",
                "source_line": "B",
                "attach_to": "car_base",
                "layer": 1,
            },
        ],
        "constraints": [
            "car_base is sourced from A and assembled on product line A.",
            "car_frame is sourced from B and transferred to product line A.",
            "car_frame is held only after it arrives on line A for final assembly.",
        ],
    },
    "phone": {
        "line": "B",
        "base_part": {"part": "phone_base", "source_line": "A"},
        "assemblies": [
            {
                "part": "screen",
                "source_line": "B",
                "attach_to": "phone_base",
                "layer": 1,
            },
            {
                "part": "camera_module",
                "source_line": "B",
                "attach_to": "phone_base",
                "layer": 2,
            },
        ],
        "constraints": [
            "phone_base is sourced from A and transferred to product line B.",
            "screen and camera_module are sourced from B.",
            "screen and camera_module are supplied by auxiliary shuttle B2 during phone assembly.",
            "screen must be installed before camera_module.",
            "supplied parts are held only during the assembly transfer.",
        ],
    },
}


SUPPORTED_STRATEGIES = {"sequential", "parallel_start"}


def build_process_knowledge_prompt() -> str:
    lines = ["Product process knowledge:"]
    for product, spec in PRODUCT_SPECS.items():
        lines.append(f"- {product}:")
        lines.append(f"  line: {spec['line']}")
        base = spec["base_part"]
        lines.append(
            f"  base_part: {base['part']} from source line {base['source_line']}"
        )
        lines.append("  assemblies:")
        for assembly in spec["assemblies"]:
            lines.append(
                "    "
                f"{assembly['part']} from source line {assembly['source_line']} "
                f"-> {assembly['attach_to']} "
                f"(layer {assembly['layer']})"
            )
        lines.append("  constraints:")
        for constraint in spec["constraints"]:
            lines.append(f"    {constraint}")
    lines.append("Strategies: sequential, parallel_start.")
    return "\n".join(lines)
