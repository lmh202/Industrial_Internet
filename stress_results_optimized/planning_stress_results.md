# Planning Stress Test Results

| prompt | architecture | ok | elapsed_s | llm_calls | llm_elapsed_s | top_actions | unique_actions | operations | tool_steps | plan_chars | source/error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 连续生产两部手机和一辆汽车 | two_layer_agent | yes | 97.19 | 3 | 97.19 | 6 | 4 | 17 | 52 | 3288 | top_planner_operation_agent_compiler |
| 按顺序生产三部手机和两辆汽车 | two_layer_agent | yes | 103.66 | 3 | 103.66 | 10 | 4 | 28 | 86 | 5312 | top_planner_operation_agent_compiler |
| 先同时启动生产一部手机和一辆汽车，完成后再生产一部手机和一辆汽车 | two_layer_agent | yes | 180.84 | 4 | 180.83 | 8 | 4 | 22 | 68 | 4212 | top_planner_operation_agent_compiler |
| 连续生产两部手机、两辆汽车，并把B线的屏幕移动到output | two_layer_agent | yes | 103.67 | 3 | 103.67 | 10 | 5 | 24 | 74 | 4517 | top_planner_operation_agent_compiler |

## Deterministic Complexity Targets

| prompt | top_actions | unique_actions | operations | compiled_tool_steps | top_json_chars | operation_json_chars | direct_tool_json_chars |
|---|---:|---:|---:|---:|---:|---:|---:|
| 连续生产两部手机和一辆汽车 | 6 | 4 | 17 | 52 | 285 | 1299 | 3204 |
| 按顺序生产三部手机和两辆汽车 | 10 | 4 | 28 | 86 | 423 | 2067 | 5227 |
| 先同时启动生产一部手机和一辆汽车，完成后再生产一部手机和一辆汽车 | 4 | 4 | 11 | 34 | 226 | 859 | 2079 |
| 连续生产两部手机、两辆汽车，并把B线的屏幕移动到output | 8 | 4 | 22 | 68 | 353 | 1616 | 4095 |
