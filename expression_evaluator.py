
import ast
import operator


# Supported mathematical operations
OPERATIONS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv
}


def evaluate_node(node):
    """Safely evaluate a mathematical expression tree."""

    if isinstance(node, ast.Expression):
        return evaluate_node(node.body)

    if isinstance(node, ast.Constant):
        if type(node.value) in (int, float):
            return node.value
        raise ValueError("Invalid number")

    if isinstance(node, ast.BinOp):
        operation = OPERATIONS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported operation")

        left = evaluate_node(node.left)
        right = evaluate_node(node.right)

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        value = evaluate_node(node.operand)

        if isinstance(node.op, ast.USub):
            return -value

        if isinstance(node.op, ast.UAdd):
            return value

    raise ValueError("Invalid expression")


def calculate_expression(expression):
    """
    Convert recognised mathematical symbols into
    Python operators and calculate the answer.
    """

    expression = expression.replace("×", "*")
    expression = expression.replace("÷", "/")
    expression = expression.replace("−", "-")
    expression = expression.replace(" ", "")

    if len(expression) > 200:
        raise ValueError("Expression is too long")

    try:
        tree = ast.parse(expression, mode="eval")
        result = evaluate_node(tree)

        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return result

    except ZeroDivisionError:
        return "Error: Division by zero"

    except (SyntaxError, ValueError, OverflowError):
        return "Error: Invalid mathematical expression"



if __name__ == "__main__":

    print("HANDWRITTEN MATHEMATICAL EXPRESSION CALCULATOR")
    print("---------------------------------------------")

    while True:

        expression = input(
            "\nEnter a mathematical expression "
            "(or type 'exit'): "
        )

        if expression.lower() == "exit":
            print("Calculator closed.")
            break

        answer = calculate_expression(expression)

        print("Expression:", expression)
        print("Answer:", answer)
