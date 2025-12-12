SYSTEM_PROMPT = """You are a helpful coding assistant that can read, write, and manage files.

You have access to the following tools:
- read_file: Read the contents of a file
- write_file: Write content to a file (creates or overwrites)
- list_files: List files in a directory

When given a task:
1. Think about what you need to do
2. Use tools to gather information or make changes
3. Continue until the task is complete
4. Explain what you did

Always be careful when writing files - make sure you understand the existing content first."""

TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path. Returns the file content as a string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read"
                }
            },
            "required": ["path"]
        }
    },

    {
        "name": "write_file",
        "description": "Write content to a file at the given path. Creates the gile if it does not exist. Can overwrite or append to existing content. Returns a status message indicating success or failure.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                },
                "mode": {
                    "type": "string",
                    "description": "The mode to write the file in: 'w' for overwrite, 'a' for append",
                    "enum": ["w", "a"],
                    "default": "w"
                }
            },
            "required": ["path", "content"]
        }
    },

    {
        "name": "list_files",
        "description": "List files in the specified directory. Can optionally include folders and filter by a simple pattern. Returns an array of file names or paths",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the directory to list files from"
                },
                "include_dirs":{
                    "type": "boolean",
                    "description": "Whether to include directories in the listing",
                    "default": False
                },           
            },
            "required": ["path"]
        }
    }    
]

MODEL = "deepseek-coder:6.7b"
    
HOST = "http://localhost:11434"


def read_file(path: str) -> str:
    """Read and return the contents of a file."""
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"