# Planning Stress Test Results

| prompt | architecture | ok | elapsed_s | llm_calls | llm_elapsed_s | top_actions | unique_actions | operations | tool_steps | plan_chars | source/error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 连续生产两部手机和一辆汽车 | two_layer_agent | yes | 204.91 | 3 | 204.91 | 6 | 4 | 17 | 52 | 3288 | top_planner_operation_agent_compiler |
| 连续生产两部手机和一辆汽车 | direct_tool_json | no | 282.85 | 1 | 282.85 | 0 | 0 | 0 | 0 | 2592 | Step 10 layer must be numeric. |
| 按顺序生产三部手机和两辆汽车 | two_layer_agent | yes | 114.52 | 3 | 114.52 | 10 | 4 | 28 | 86 | 5312 | top_planner_operation_agent_compiler |
| 按顺序生产三部手机和两辆汽车 | direct_tool_json | no | 366.23 | 1 | 366.23 | 0 | 0 | 0 | 0 | 4446 | Step 10 layer must be numeric. |

## Deterministic Complexity Targets

| prompt | top_actions | unique_actions | operations | compiled_tool_steps | top_json_chars | operation_json_chars | direct_tool_json_chars |
|---|---:|---:|---:|---:|---:|---:|---:|
| 连续生产两部手机和一辆汽车 | 6 | 4 | 17 | 52 | 285 | 1299 | 3204 |
| 按顺序生产三部手机和两辆汽车 | 10 | 4 | 28 | 86 | 423 | 2067 | 5227 |
