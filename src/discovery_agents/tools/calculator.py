"""A safe arithmetic calculator tool (no eval of arbitrary code)."""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable

from .base import ToolResult

# Only these AST node types / operators are permitted.
_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError("unsupported expression")


class CalculatorTool:
    """Evaluate a basic arithmetic expression safely."""

    name = "calculator"
    description = "Evaluate a basic arithmetic expression (+, -, *, /, //, %, **)."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "e.g. '2 + 2 * 3'"}},
        "required": ["expression"],
    }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        expression = str(arguments.get("expression", "")).strip()
        if not expression:
            return ToolResult(ok=False, error="expression is required")
        try:
            tree = ast.parse(expression, mode="eval")
            result = _evaluate(tree.body)
        except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as exc:
            return ToolResult(ok=False, error=f"cannot evaluate: {exc}")
        return ToolResult(ok=True, data={"expression": expression, "result": result})
