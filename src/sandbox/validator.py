"""
Слой 1 среды безопасного исполнения: статический AST-анализ кода.

Перед выполнением код проверяется через ast.parse() и обход дерева.
Опасные паттерны отклоняются до запуска любого процесса.
"""
import ast
from typing import Tuple, List


# Модули, импорт которых запрещён.
# Docker уже обеспечивает изоляцию (--read-only, --network=none, nobody, --cap-drop=ALL),
# поэтому блокируем только то, что опасно ВНУТРИ контейнера: спавн процессов,
# низкоуровневый доступ к памяти/ядру и сериализацию с выполнением кода.
BLOCKED_IMPORTS: frozenset = frozenset({
    # Спавн процессов и shell-команды — опасны даже в контейнере
    "subprocess", "multiprocessing", "os",
    # Сеть — закрыта --network=none, но блокируем и на уровне AST
    "socket", "requests", "urllib", "http", "ftplib", "smtplib",
    "xmlrpc", "imaplib", "poplib", "telnetlib", "ssl",
    # Низкоуровневый доступ к памяти и ядру
    "ctypes", "cffi",
    # Динамический импорт и интроспекция кода
    "importlib", "builtins",
    # Сериализация, выполняющая произвольный код при десериализации
    "pickle", "marshal", "shelve", "dill", "cloudpickle",
    # Windows-специфичные системные модули
    "winreg", "msvcrt", "nt", "_winapi",
    # Инструменты выполнения/компиляции кода
    "code", "codeop", "py_compile", "compileall",
})

# Встроенные функции, вызов которых запрещён.
# open() убрана: Docker --read-only + отсутствие volume-монтирований уже защищают FS.
BLOCKED_BUILTINS: frozenset = frozenset({
    "exec", "eval", "compile",   # выполнение произвольного кода
    "input",                     # интерактивный ввод (зависнет в subprocess)
    "__import__",                 # динамический импорт
    "breakpoint",                # вызывает отладчик
    "memoryview",                # прямой доступ к буферам памяти
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
