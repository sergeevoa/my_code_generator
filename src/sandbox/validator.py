"""
Слой 1 песочницы: статический AST-анализ кода.

Перед выполнением каждый код проверяется через ast.parse() и обход дерева.
Опасные паттерны отклоняются до запуска любого процесса.
"""
import ast
from typing import Tuple, List


# Модули, импорт которых запрещён
BLOCKED_IMPORTS: frozenset = frozenset({
    # Файловая система и процессы
    "os", "subprocess", "shutil", "pathlib", "glob", "fnmatch",
    # Сеть
    "socket", "requests", "urllib", "http", "ftplib", "smtplib",
    "xmlrpc", "imaplib", "poplib", "telnetlib", "ssl",
    # Системные/низкоуровневые
    "sys", "ctypes", "cffi", "io", "builtins", "importlib",
    "multiprocessing", "threading", "asyncio", "concurrent",
    "signal", "resource", "gc",
    # Сериализация (могут выполнять произвольный код)
    "pickle", "marshal", "shelve", "dill", "cloudpickle",
    # Windows-специфичные
    "winreg", "msvcrt", "nt", "_winapi", "winsound",
    # Прочие потенциально опасные
    "code", "codeop", "pdb", "profile", "cProfile", "traceback",
    "inspect", "dis", "py_compile", "compileall",
})

# Встроенные функции, вызов которых запрещён
BLOCKED_BUILTINS: frozenset = frozenset({
    "exec", "eval", "compile",           # выполнение произвольного кода
    "open",                               # доступ к файловой системе
    "input",                              # интерактивный ввод (зависнет в subprocess)
    "__import__",                         # динамический импорт
    "getattr", "setattr", "delattr",      # обход атрибутов через строки
    "globals", "locals", "vars",          # доступ к пространству имён
    "breakpoint",                         # отладчик
    "memoryview",                         # прямой доступ к памяти
    "super",                              # обход MRO для дандер-методов
})

# Имена атрибутов, обращение к которым запрещено
BLOCKED_ATTRIBUTES: frozenset = frozenset({
    "__code__", "__globals__", "__builtins__",
    "__subclasses__", "__bases__", "__mro__",
    "__class__", "__dict__", "__module__",
    "__reduce__", "__reduce_ex__", "__init_subclass__",
    "__loader__", "__spec__", "__file__",
})


class SafetyValidator(ast.NodeVisitor):
    """
    Обходит AST-дерево и собирает нарушения безопасности.
    Нарушения не прерывают обход — собираются все сразу.
    """

    def __init__(self) -> None:
        self.errors: List[str] = []

    # --- Импорты ---

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_module = alias.name.split(".")[0]
            if base_module in BLOCKED_IMPORTS:
                self.errors.append(
                    f"Строка {node.lineno}: запрещённый импорт '{alias.name}'"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base_module = node.module.split(".")[0]
            if base_module in BLOCKED_IMPORTS:
                self.errors.append(
                    f"Строка {node.lineno}: запрещённый импорт 'from {node.module} import ...'"
                )
        self.generic_visit(node)

    # --- Вызовы функций ---

    def visit_Call(self, node: ast.Call) -> None:
        # Прямой вызов: exec("..."), eval("..."), open(...)
        if isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_BUILTINS:
                self.errors.append(
                    f"Строка {node.lineno}: запрещённый вызов '{node.func.id}()'"
                )
        # Вызов через атрибут: obj.__class__(), type.__subclasses__() и т.д.
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in BLOCKED_ATTRIBUTES:
                self.errors.append(
                    f"Строка {node.lineno}: запрещённый вызов метода '.{node.func.attr}()'"
                )
        self.generic_visit(node)

    # --- Доступ к атрибутам ---

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in BLOCKED_ATTRIBUTES:
            self.errors.append(
                f"Строка {node.lineno}: запрещённый доступ к атрибуту '.{node.attr}'"
            )
        self.generic_visit(node)


def validate_code(code: str) -> Tuple[bool, List[str]]:
    """
    Проверяет Python-код на безопасность.

    Returns:
        (True, [])               — код безопасен
        (False, ["...", ...])    — список найденных нарушений
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Синтаксическая ошибка: {e}"]

    validator = SafetyValidator()
    validator.visit(tree)

    is_safe = len(validator.errors) == 0
    return is_safe, validator.errors
