"""Keep the PythonREPL calculator, but admit only bounded arithmetic expressions."""
import ast
from langchain_experimental.tools import PythonREPLTool

_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub,
            ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.UAdd, ast.USub)


def calculate(expression: str) -> str:
    try:
        expression = expression.strip()
        # Accommodate ReAct's common print(...) form without allowing arbitrary calls.
        if expression.startswith('print(') and expression.endswith(')'):
            expression = expression[6:-1]
        if len(expression) > 500:
            raise ValueError
        tree = ast.parse(expression, mode='eval')
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED):
                raise ValueError
            if isinstance(node, ast.Constant) and (type(node.value) not in (int, float) or abs(node.value) > 1e100):
                raise ValueError
        return PythonREPLTool().run('print(' + expression + ')')
    except Exception:
        return 'Calculator accepts numeric arithmetic with +, -, *, /, //, %, and parentheses only.'
