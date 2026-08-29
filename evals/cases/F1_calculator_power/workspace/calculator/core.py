"""Calculator operations."""


def add(left: float, right: float) -> float:
    return left + right


def subtract(left: float, right: float) -> float:
    return left - right


def multiply(left: float, right: float) -> float:
    return left * right


def divide(left: float, right: float) -> float:
    if right == 0:
        raise ValueError("Cannot divide by zero")
    return left / right


OPERATIONS = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
}


def calculate(operation: str, left: float, right: float) -> float:
    try:
        function = OPERATIONS[operation]
    except KeyError as error:
        raise ValueError(f"Unknown operation: {operation}") from error
    return function(left, right)
