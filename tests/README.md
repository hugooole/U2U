<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2025 zhiguo -->

# UsdParser 测试框架

这是一个可扩展的测试框架，用于测试 `UsdParser` 解析各种类型的 USD 文件。

## 测试结构

```
tests/
├── test_usd_parser.py          # 主测试文件
├── usd_parser_helpers.py       # 辅助函数模块
├── usd_test_cases.yaml         # 测试用例配置
└── README.md                   # 本文档
```

## 运行测试

```bash
# 运行所有 UsdParser 测试
uv run pytest tests/test_usd_parser.py -v

# 运行特定测试类
uv run pytest tests/test_usd_parser.py::TestUsdParserRigidBody -v

# 运行特定测试用例
uv run pytest tests/test_usd_parser.py -k "cartpole" -v

# 运行所有测试
uv run pytest tests/ -v
```

## 测试覆盖范围

### 当前测试类型

1. **刚体解析** (`TestUsdParserRigidBody`)
   - 基本解析功能
   - 与 USD stage 的一致性验证

2. **关节机器人解析** (`TestUsdParserArticulation`)
   - 关节结构解析
   - 关节类型和限制
   - 与 USD stage 的一致性验证

3. **工具测试** (`TestUsdParserUtils`)
   - 单位转换
   - up-axis 检测
   - Scene 初始化

4. **边界情况** (`TestUsdParserEdgeCases`)
   - 空场景处理
   - 配置传播

### 当前测试文件

- `assets/usd/AnalyticCone.usda` - 简单刚体
- `assets/usd/HelloWorldSim.usda` - 多刚体场景
- `assets/usd/cartpole.usda` - 关节系统
- `assets/usd/prismatic_joint.usda` - 棱柱关节

## 添加新测试用例

### 步骤 1: 准备 USD 文件

将新的 USD 文件放入 `assets/usd/` 目录。

### 步骤 2: 更新配置文件

编辑 `tests/usd_test_cases.yaml`，添加新的测试用例：

```yaml
rigid_body_tests:
  - name: "my_new_test"
    file: "assets/usd/my_new_file.usda"
    description: "测试描述"
    expected:
      min_geometries: 1
      has_rigid_bodies: true
```

### 步骤 3: 运行测试

```bash
uv run pytest tests/test_usd_parser.py -v
```

测试框架会自动为新配置生成测试用例。

## 配置选项

### 刚体测试 (`rigid_body_tests`)

```yaml
expected:
  min_geometries: 1          # 最小几何体数量
  num_geometries: 2          # 精确几何体数量（可选）
  has_rigid_bodies: true     # 是否包含刚体
```

### 关节测试 (`articulation_tests`)

```yaml
expected:
  min_robots: 1              # 最小机器人数量
  num_robots: 1              # 精确机器人数量（可选）
  has_joints: true           # 是否包含关节
```

## 辅助函数

`usd_parser_helpers.py` 提供以下辅助函数：

- `load_usd_stage(usd_path)` - 加载 USD stage
- `count_prims_with_api(stage, api_schema_name)` - 统计包含特定 API 的 prim 数量
- `parse_usd_with_parser(usd_path, config)` - 使用 UsdParser 解析 USD 文件
- `extract_joint_info(stage)` - 从 USD stage 提取关节信息
- `validate_geometry_dict(geometry_dict, expected)` - 验证 geometry_dict
- `validate_robot_dict(robot_dict, expected)` - 验证 robot_dict

## 注意事项

1. **Deformable 和 Cloth**: 这些类型通常在代码中动态应用 API，而不是在 USD 文件中预定义，因此不包含在测试中。

2. **Mesh Approximation**: 由于 mesh approximation（如 convex decomposition）可能生成多个凸包，解析的几何体数量可能多于 USD stage 中的原始数量。

3. **测试隔离**: 每个测试都是独立的，不会运行 `world.advance()` 或 Polyscope 可视化，保持测试轻量快速。

## 扩展测试框架

### 添加新的测试类

在 `test_usd_parser.py` 中添加新的测试类：

```python
class TestUsdParserNewFeature:
    """新功能测试。"""

    @pytest.mark.parametrize("test_case", load_test_cases()["new_feature_tests"])
    def test_new_feature(self, test_case):
        """测试新功能。"""
        scene = parse_usd_with_parser(test_case["file"])
        # 添加验证逻辑
        assert scene is not None
```

### 添加新的辅助函数

在 `usd_parser_helpers.py` 中添加新的辅助函数：

```python
def my_helper_function(stage: Usd.Stage) -> dict:
    """辅助函数描述。"""
    # 实现逻辑
    return result
```

## 测试统计

当前测试数量：**16 个测试**

- 刚体测试: 4
- 关节测试: 6
- 工具测试: 4
- 边界情况: 2
