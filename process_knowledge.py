"""Product and process knowledge for high-level production planning."""

from __future__ import annotations


PRODUCT_SPECS = {
    "car": {
        "line": "A",
        "base_part": "car_base",
        "assemblies": [
            {"part": "car_frame", "attach_to": "car_base", "layer": 1},
        ],
        "constraints": [
            "car_base stays on the main A shuttle.",
            "car_frame is supplied by auxiliary shuttle A2.",
            "Robot_Assemble_Car should only hold car_frame briefly during transfer.",
        ],
    },
    "phone": {
        "line": "B",
        "base_part": "phone_base",
        "assemblies": [
            {"part": "screen", "attach_to": "phone_base", "layer": 1},
            {"part": "camera_module", "attach_to": "phone_base", "layer": 2},
        ],
        "constraints": [
            "phone_base stays on the main B shuttle.",
            "screen and camera_module are supplied by auxiliary shuttle B2.",
            "screen must be installed before camera_module.",
            "Robot_Assemble_Phone should only hold supplied parts briefly.",
        ],
    },
}


SUPPORTED_STRATEGIES = {"sequential", "parallel_start"}


def build_process_knowledge_prompt() -> str:
    lines = ["Product process knowledge:"]
    for product, spec in PRODUCT_SPECS.items():
        lines.append(f"- {product}:")
        lines.append(f"  line: {spec['line']}")
        lines.append(f"  base_part: {spec['base_part']}")
        lines.append("  assemblies:")
        for assembly in spec["assemblies"]:
            lines.append(
                "    "
                f"{assembly['part']} -> {assembly['attach_to']} "
                f"(layer {assembly['layer']})"
            )
        lines.append("  constraints:")
        for constraint in spec["constraints"]:
            lines.append(f"    {constraint}")
    lines.append("Strategies: sequential, parallel_start.")
    return "\n".join(lines)
