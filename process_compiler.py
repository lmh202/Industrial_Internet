"""Compile high-level process plans into executable tool-call plans."""

from __future__ import annotations

from typing import Any

from process_knowledge import PRODUCT_SPECS, SUPPORTED_STRATEGIES
from tool_registry import SUPPORTED_PARTS
from validator import PlanValidationError, validate_plan, validate_plan_sequence


LINE_PARTS = {
    "A": {"car_base", "car_frame"},
    "B": {"phone_base", "screen", "camera_module"},
}

DEFAULT_LINE_PART = {
    "A": "car_base",
    "B": "phone_base",
}


def validate_process_plan(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PlanValidationError("Process plan must be a JSON object.")

    plan_name = payload.get("plan_name")
    if not isinstance(plan_name, str) or not plan_name.strip():
        raise PlanValidationError("Process plan must include a non-empty plan_name.")

    strategy = payload.get("strategy", "sequential")
    if not isinstance(strategy, str) or strategy not in SUPPORTED_STRATEGIES:
        raise PlanValidationError(f"Unsupported process strategy: {strategy!r}")

    operations = payload.get("operations")
    actions = payload.get("actions")
    jobs = payload.get("jobs")
    populated = [
        name
        for name, value in (
            ("operations", operations),
            ("actions", actions),
            ("jobs", jobs),
        )
        if value is not None
    ]
    if len(populated) > 1:
        raise PlanValidationError(
            "Process plan must use exactly one of operations, actions, or jobs."
        )

    clean: dict[str, Any] = {
        "plan_name": plan_name.strip(),
        "strategy": strategy,
    }
    if operations is not None:
        if not isinstance(operations, list) or not operations:
            raise PlanValidationError("Process operations must be a non-empty list.")
        clean["operations"] = [
            _validate_operation(index, operation)
            for index, operation in enumerate(operations, start=1)
        ]
    elif actions is not None:
        if not isinstance(actions, list) or not actions:
            raise PlanValidationError("Process actions must be a non-empty list.")
        clean["actions"] = [
            _validate_action(index, action)
            for index, action in enumerate(actions, start=1)
        ]
    else:
        if not isinstance(jobs, list) or not jobs:
            raise PlanValidationError(
                "Process plan must include a non-empty operations, actions, or jobs list."
            )
        clean["jobs"] = [
            _validate_job(index, job)
            for index, job in enumerate(jobs, start=1)
        ]

    assumptions = payload.get("assumptions")
    if assumptions is not None:
        if not isinstance(assumptions, list) or not all(
            isinstance(item, str) and item.strip() for item in assumptions
        ):
            raise PlanValidationError("Process assumptions must be non-empty strings.")
        clean["assumptions"] = [item.strip() for item in assumptions]
    return clean


def compile_process_plan(process_plan: dict[str, Any]) -> dict[str, Any]:
    process_plan = validate_process_plan(process_plan)
    strategy = process_plan["strategy"]

    if "operations" in process_plan:
        operations = process_plan["operations"]
    elif "actions" in process_plan:
        operations = actions_to_operations(process_plan["actions"], strategy)
    else:
        operations = jobs_to_operations(process_plan["jobs"], strategy)
    steps = compile_process_operations(operations, strategy)

    plan: dict[str, Any] = {
        "plan_name": process_plan["plan_name"],
        "steps": steps,
    }
    if "assumptions" in process_plan:
        plan["assumptions"] = process_plan["assumptions"]
    plan = validate_plan(plan)
    validate_plan_sequence(plan)
    return plan


def compile_process_operations(
    operations: list[dict[str, Any]],
    strategy: str = "sequential",
) -> list[dict[str, Any]]:
    clean_operations = [
        _validate_operation(index, operation)
        for index, operation in enumerate(operations, start=1)
    ]
    if strategy == "parallel_start":
        clean_operations = _interleave_operations_by_line(clean_operations)

    steps: list[dict[str, Any]] = []
    line_started = {"A": False, "B": False}
    line_output = {"A": False, "B": False}
    for operation in clean_operations:
        op = operation["op"]
        line = operation.get("line")
        if op == "load_base":
            if line_output[line]:
                raise PlanValidationError(
                    f"Line {line} is at output; reset_status is required before load_base."
                )
            steps.extend(_compile_load_base(operation))
            line_started[line] = True
        elif op == "assemble":
            if not line_started[line]:
                raise PlanValidationError(
                    f"Cannot assemble on line {line} before load_base."
                )
            steps.extend(_compile_assemble(operation))
        elif op == "inspect":
            steps.extend(_compile_inspect(operation))
        elif op == "unload":
            steps.extend(_compile_unload(operation))
            line_output[line] = True
            line_started[line] = False
        elif op == "move_to_output":
            if line_output[line]:
                raise PlanValidationError(
                    f"Line {line} is at output; reset_status is required before move_to_output."
                )
            steps.extend(_move_part_to_output_steps(operation["line"], operation["part"]))
            line_output[line] = True
            line_started[line] = False
        elif op == "reset_status":
            if not line_output[line]:
                raise PlanValidationError(
                    f"Cannot reset line {line} because it is not at output."
                )
            steps.append(_return_line_to_pick_step(line))
            line_output[line] = False
            line_started[line] = False
        else:
            raise PlanValidationError(f"Unsupported process operation: {op}")
    return steps


def actions_to_operations(
    actions: list[dict[str, Any]],
    strategy: str = "sequential",
) -> list[dict[str, Any]]:
    clean_actions = [
        _validate_action(index, action)
        for index, action in enumerate(actions, start=1)
    ]
    if strategy == "parallel_start" and _has_car_and_phone_actions(clean_actions):
        a_ops: list[dict[str, Any]] = []
        b_ops: list[dict[str, Any]] = []
        for action in clean_actions:
            target = a_ops if _line_for_action(action) == "A" else b_ops
            target.extend(_action_to_operations(action))
        return _interleave_operations(a_ops, b_ops)

    operations: list[dict[str, Any]] = []
    for action in clean_actions:
        operations.extend(_action_to_operations(action))
    return operations


def jobs_to_operations(
    jobs: list[dict[str, Any]],
    strategy: str = "sequential",
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    line_has_output = {"A": False, "B": False}
    for job in jobs:
        if job["action"] == "produce":
            product = job["product"]
            line = PRODUCT_SPECS[product]["line"]
            for _ in range(job["quantity"]):
                if line_has_output[line]:
                    actions.append({"action": "reset_status", "line": line})
                    line_has_output[line] = False
                actions.append({"action": f"produce_{product}"})
                line_has_output[line] = True
        else:
            line = job["line"]
            if line_has_output[line]:
                actions.append({"action": "reset_status", "line": line})
                line_has_output[line] = False
            actions.append({
                "action": "move_to_output",
                "line": line,
                "part": job["part"],
            })
            line_has_output[line] = True
    return actions_to_operations(actions, strategy)


def _validate_job(index: int, job: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(job, dict):
        raise PlanValidationError(f"Process job {index} must be an object.")
    action = job.get("action", "produce")
    if action == "produce":
        product = job.get("product")
        if product not in PRODUCT_SPECS:
            raise PlanValidationError(
                f"Process job {index} has unsupported product: {product!r}"
            )
        quantity = job.get("quantity", 1)
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            raise PlanValidationError(
                f"Process job {index} quantity must be a positive integer."
            )
        return {"action": "produce", "product": product, "quantity": quantity}
    if action == "move_to_output":
        return _validate_move_target(index, job)
    raise PlanValidationError(f"Process job {index} has unsupported action: {action!r}")


def _validate_action(index: int, action: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(action, dict):
        raise PlanValidationError(f"Process action {index} must be an object.")
    action_name = action.get("action")
    if action_name in {"produce_car", "produce_phone"}:
        return {"action": action_name}
    if action_name == "reset_status":
        line = action.get("line")
        if line not in LINE_PARTS:
            raise PlanValidationError(
                f"Process action {index} reset_status has invalid line: {line!r}"
            )
        return {"action": "reset_status", "line": line}
    if action_name == "move_to_output":
        return _validate_move_target(index, action)
    raise PlanValidationError(
        f"Process action {index} has unsupported action: {action_name!r}"
    )


def _validate_move_target(index: int, payload: dict[str, Any]) -> dict[str, Any]:
    line = payload.get("line")
    if line not in LINE_PARTS:
        raise PlanValidationError(f"Process item {index} has unsupported line: {line!r}")
    part = payload.get("part") or DEFAULT_LINE_PART[line]
    if part not in SUPPORTED_PARTS or part not in LINE_PARTS[line]:
        raise PlanValidationError(
            f"Process item {index} cannot move part {part!r} on line {line}."
        )
    return {"action": "move_to_output", "line": line, "part": part}


def _validate_operation(index: int, operation: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(operation, dict):
        raise PlanValidationError(f"Process operation {index} must be an object.")
    op = operation.get("op")
    if op == "load_base":
        line = _required_line(index, operation)
        part = _required_part(index, operation, "part")
        if part not in LINE_PARTS[line]:
            raise PlanValidationError(
                f"Process operation {index} cannot load {part} on line {line}."
            )
        return {"op": "load_base", "line": line, "part": part}
    if op == "assemble":
        line = _required_line(index, operation)
        base = _required_part(index, operation, "base")
        part = _required_part(index, operation, "part")
        supplier = operation.get("supplier")
        expected_supplier = "A2" if line == "A" else "B2"
        if supplier != expected_supplier:
            raise PlanValidationError(
                f"Process operation {index} line {line} requires supplier {expected_supplier}."
            )
        if base not in LINE_PARTS[line] or part not in LINE_PARTS[line]:
            raise PlanValidationError(
                f"Process operation {index} has part not available on line {line}."
            )
        layer = operation.get("layer", 1)
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise PlanValidationError(f"Process operation {index} layer must be an integer.")
        return {
            "op": "assemble",
            "line": line,
            "base": base,
            "part": part,
            "supplier": supplier,
            "layer": layer,
        }
    if op == "inspect":
        return {"op": "inspect", "line": _required_line(index, operation)}
    if op == "unload":
        return {
            "op": "unload",
            "line": _required_line(index, operation),
            "part": _required_part(index, operation, "part"),
        }
    if op == "move_to_output":
        line = _required_line(index, operation)
        part = _required_part(index, operation, "part")
        if part not in LINE_PARTS[line]:
            raise PlanValidationError(
                f"Process operation {index} cannot move {part} on line {line}."
            )
        return {"op": "move_to_output", "line": line, "part": part}
    if op == "reset_status":
        return {"op": "reset_status", "line": _required_line(index, operation)}
    raise PlanValidationError(f"Process operation {index} has unsupported op: {op!r}")


def _required_line(index: int, operation: dict[str, Any]) -> str:
    line = operation.get("line")
    if line not in LINE_PARTS:
        raise PlanValidationError(f"Process operation {index} has invalid line: {line!r}")
    return line


def _required_part(index: int, operation: dict[str, Any], field: str) -> str:
    part = operation.get(field)
    if part not in SUPPORTED_PARTS:
        raise PlanValidationError(
            f"Process operation {index} has invalid {field}: {part!r}"
        )
    return part


def _line_for_action(action: dict[str, Any]) -> str:
    action_name = action["action"]
    if action_name == "produce_car":
        return "A"
    if action_name == "produce_phone":
        return "B"
    return action["line"]


def _action_to_operations(action: dict[str, Any]) -> list[dict[str, Any]]:
    action_name = action["action"]
    if action_name == "produce_car":
        return _product_operations("car")
    if action_name == "produce_phone":
        return _product_operations("phone")
    if action_name == "reset_status":
        return [{"op": "reset_status", "line": action["line"]}]
    if action_name == "move_to_output":
        return [{
            "op": "move_to_output",
            "line": action["line"],
            "part": action["part"],
        }]
    raise PlanValidationError(f"Unsupported process action: {action_name}")


def _product_operations(product: str) -> list[dict[str, Any]]:
    spec = PRODUCT_SPECS[product]
    line = spec["line"]
    base = spec["base_part"]
    supplier = "A2" if line == "A" else "B2"
    operations: list[dict[str, Any]] = [
        {"op": "load_base", "line": line, "part": base},
    ]
    for assembly in spec["assemblies"]:
        operations.append({
            "op": "assemble",
            "line": line,
            "base": base,
            "part": assembly["part"],
            "supplier": supplier,
            "layer": assembly["layer"],
        })
    operations.extend([
        {"op": "inspect", "line": line},
        {"op": "unload", "line": line, "part": base},
    ])
    return operations


def _interleave_operations_by_line(
    operations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    a_ops = [operation for operation in operations if operation.get("line") == "A"]
    b_ops = [operation for operation in operations if operation.get("line") == "B"]
    other_ops = [
        operation for operation in operations
        if operation.get("line") not in {"A", "B"}
    ]
    return _interleave_operations(a_ops, b_ops) + other_ops


def _interleave_operations(
    a_operations: list[dict[str, Any]],
    b_operations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    max_len = max(len(a_operations), len(b_operations))
    for index in range(max_len):
        if index < len(a_operations):
            operations.append(a_operations[index])
        if index < len(b_operations):
            operations.append(b_operations[index])
    return operations


def _has_car_and_phone_actions(actions: list[dict[str, Any]]) -> bool:
    action_names = {action["action"] for action in actions}
    return {"produce_car", "produce_phone"}.issubset(action_names)


def _compile_load_base(operation: dict[str, Any]) -> list[dict[str, Any]]:
    line = operation["line"]
    if line == "A":
        return [
            {"tool": "Load_A_Pick", "args": {"part": operation["part"]}},
            {"tool": "Transport_A_Pick_Assemble", "args": {}},
        ]
    return [
        {"tool": "Load_B_Pick", "args": {"part": operation["part"]}},
        {"tool": "Transport_B_Pick_Assemble", "args": {}},
    ]


def _compile_assemble(operation: dict[str, Any]) -> list[dict[str, Any]]:
    line = operation["line"]
    part = operation["part"]
    base = operation["base"]
    layer = operation["layer"]
    if line == "A":
        return [
            {"tool": "Transport_A_Assemble_Forward", "args": {}},
            {"tool": "Transport_A2_Clear_Pick", "args": {}},
            {"tool": "Load_A2_Pick", "args": {"part": part}},
            {"tool": "Transport_A2_Pick_Assemble", "args": {}},
            {"tool": "Hold_A2_Assemble", "args": {"part": part}},
            {"tool": "Transport_A2_Assemble_Clear", "args": {}},
            {"tool": "Transport_A_Forward_Assemble", "args": {}},
            {
                "tool": "Place_A_Assemble",
                "args": {"part": part, "attach_to": base, "layer": layer},
            },
        ]
    return [
        {"tool": "Transport_B_Assemble_Forward", "args": {}},
        {"tool": "Transport_B2_Clear_Pick", "args": {}},
        {"tool": "Load_B2_Pick", "args": {"part": part}},
        {"tool": "Transport_B2_Pick_Assemble", "args": {}},
        {"tool": "Hold_B2_Assemble", "args": {"part": part}},
        {"tool": "Transport_B2_Assemble_Clear", "args": {}},
        {"tool": "Transport_B_Forward_Assemble", "args": {}},
        {
            "tool": "Place_B_Assemble",
            "args": {"part": part, "attach_to": base, "layer": layer},
        },
    ]


def _compile_inspect(operation: dict[str, Any]) -> list[dict[str, Any]]:
    if operation["line"] == "A":
        return [
            {"tool": "Transport_A_Assemble_Camera", "args": {}},
            {"tool": "Inspect_A", "args": {}},
        ]
    return [
        {"tool": "Transport_B_Assemble_Camera", "args": {}},
        {"tool": "Inspect_B", "args": {}},
    ]


def _compile_unload(operation: dict[str, Any]) -> list[dict[str, Any]]:
    if operation["line"] == "A":
        return [
            {"tool": "Transport_A_Camera_Output", "args": {}},
            {"tool": "Unload_A_Output", "args": {"part": operation["part"]}},
        ]
    return [
        {"tool": "Transport_B_Camera_Output", "args": {}},
        {"tool": "Unload_B_Output", "args": {"part": operation["part"]}},
    ]


def _return_line_to_pick_step(line: str) -> dict[str, Any]:
    if line == "A":
        return {"tool": "Transport_A_Output_Pick", "args": {}}
    return {"tool": "Transport_B_Output_Pick", "args": {}}


def _move_part_to_output_steps(line: str, part: str) -> list[dict[str, Any]]:
    if line == "A":
        return [
            {"tool": "Load_A_Pick", "args": {"part": part}},
            {"tool": "Transport_A_Pick_Assemble", "args": {}},
            {"tool": "Transport_A_Assemble_Camera", "args": {}},
            {"tool": "Transport_A_Camera_Output", "args": {}},
            {"tool": "Unload_A_Output", "args": {"part": part}},
        ]
    if part == "screen":
        return [
            {"tool": "Transport_B_Pick_Assemble", "args": {}},
            {"tool": "Transport_B2_Clear_Pick", "args": {}},
            {"tool": "Load_B2_Pick", "args": {"part": "screen"}},
            {"tool": "Transport_B2_Pick_Assemble", "args": {}},
            {"tool": "Hold_B2_Assemble", "args": {"part": "screen"}},
            {"tool": "Transport_B2_Assemble_Clear", "args": {}},
            {"tool": "Place_B_Assemble", "args": {"part": "screen", "layer": 0}},
            {"tool": "Transport_B_Assemble_Camera", "args": {}},
            {"tool": "Transport_B_Camera_Output", "args": {}},
            {"tool": "Unload_B_Output", "args": {"part": "screen"}},
        ]
    return [
        {"tool": "Load_B_Pick", "args": {"part": part}},
        {"tool": "Transport_B_Pick_Assemble", "args": {}},
        {"tool": "Transport_B_Assemble_Camera", "args": {}},
        {"tool": "Transport_B_Camera_Output", "args": {}},
        {"tool": "Unload_B_Output", "args": {"part": part}},
    ]
