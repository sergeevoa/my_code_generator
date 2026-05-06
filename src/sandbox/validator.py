"""
Слой 1 среды безопасного исполнения: статический AST-анализ кода.

Перед выполнением код проверяется через ast.parse() и обход дерева.
Опасные паттерны отклоняются до запуска любого процесса.

Две независимые проверки (вызываются в этом порядке):
  validate_code()               — код содержит опасные конструкции (угроза безопасности)
  check_sandbox_compatibility() — код требует сети/БД/сервера (ограничение среды)
"""
import ast
from typing import Tuple, List


# Модули, импорт которых запрещён по соображениям безопасности.
BLOCKED_IMPORTS: frozenset = frozenset({
    # Спавн процессов и shell-команды — опасны даже в контейнере
    "subprocess", "multiprocessing", "os",
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

# ── Модули, несовместимые с изолированным sandbox ──────────────────────────
# Не опасны сами по себе, но требуют внешней инфраструктуры.

NETWORK_MODULES: frozenset = frozenset({
    # HTTP-клиенты
    "requests", "httpx", "aiohttp", "urllib3", "urllib", "http", "httplib2",
    # WebSocket
    "websockets", "websocket", "socketio",
    # Низкоуровневая сеть
    "socket", "ssl",
    # Прикладные протоколы
    "ftplib", "smtplib", "imaplib", "poplib", "telnetlib", "xmlrpc",
})

DATABASE_MODULES: frozenset = frozenset({
    # ORM и миграции
    "sqlalchemy", "alembic", "tortoise", "databases",
    # PostgreSQL
    "psycopg2", "psycopg", "asyncpg",
    # MySQL
    "pymysql", "aiomysql", "MySQLdb", "mysql",
    # MongoDB
    "pymongo", "motor", "mongoengine",
    # Redis
    "redis", "aioredis",
    # Прочие
    "pyodbc", "cx_Oracle", "aioodbc", "elasticsearch", "cassandra",
    # sqlite3 намеренно не включён — работает локально без сервера
})

SERVER_MODULES: frozenset = frozenset({
    # Веб-фреймворки
    "flask", "fastapi", "django", "tornado", "starlette",
    "sanic", "falcon", "bottle", "cherrypy", "quart",
    # ASGI/WSGI-серверы
    "uvicorn", "gunicorn", "hypercorn", "daphne", "waitress",
    # Брокеры сообщений и очереди
    "pika", "kafka", "celery", "kombu", "aio_pika", "nats",
    # RPC
    "grpc",
})

_INCOMPATIBLE_CATEGORIES: tuple = (
    ("сетевые библиотеки (сеть недоступна: --network=none)",        NETWORK_MODULES),
    ("драйверы баз данных (нет подключения к серверу БД)",           DATABASE_MODULES),
    ("веб-фреймворки / брокеры (некуда биндить порт или очередь)",  SERVER_MODULES),
)

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


def check_sandbox_compatibility(code: str) -> Tuple[bool, str]:
    """
    Проверяет, совместим ли код с изолированным sandbox.

    Вызывается ПОСЛЕ validate_code — только для кода, прошедшего проверку безопасности.

    Returns:
        (True,  "")             — код можно запустить в sandbox
        (False, "<сообщение>")  — код требует сети/БД/сервера
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True, ""  # синтаксическую ошибку обработает executor

    found: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module.split(".")[0]] if node.module else []
        else:
            continue

        for name in names:
            for reason, module_set in _INCOMPATIBLE_CATEGORIES:
                if name in module_set:
                    found.setdefault(reason, set()).add(name)

    if not found:
        return True, ""

    details = "\n".join(
        f"  • {reason}: {', '.join(sorted(mods))}"
        for reason, mods in found.items()
    )
    message = (
        "[NOT TESTABLE] Код не может быть выполнен в изолированной среде.\n"
        "Обнаружены модули, требующие внешней инфраструктуры:\n"
        f"{details}\n\n"
        "Sandbox работает без сетевого доступа и без внешних сервисов.\n"
        "Вынесите логику, не зависящую от сети/БД/сервера, в отдельные функции — "
        "их можно протестировать локально."
    )
    return False, message


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
