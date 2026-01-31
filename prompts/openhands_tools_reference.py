"""
OpenHands SDK Tool Implementations for DGM Self-Improvement Reference

This module contains simplified implementations of OpenHands SDK tools
that can be used as references in prompts to help DGM self-evolve.
"""

# ============================================================================
# 1. String Replace Tool (str_replace) - MOST RECOMMENDED
# ============================================================================

STR_REPLACE_IMPLEMENTATION = """
def str_replace(path: str, old_str: str, new_str: str) -> str:
    '''
    Replace old_str with new_str in the file at path.
    Only replaces if old_str appears exactly once in the file.
    
    This is more efficient than overwriting the entire file, especially for
    large files where only a small change is needed.
    
    Args:
        path: Absolute path to the file
        old_str: String to replace (must appear exactly once)
        new_str: Replacement string
    
    Returns:
        Success message with snippet of edited section, or error message
    '''
    from pathlib import Path
    import re
    
    path_obj = Path(path)
    
    # Validate file exists
    if not path_obj.exists() or not path_obj.is_file():
        return f"Error: File {path} does not exist or is not a file."
    
    # Read file content
    try:
        file_content = path_obj.read_text()
    except Exception as e:
        return f"Error: Failed to read file: {e}"
    
    # Find all occurrences using regex (escape special chars to match literally)
    pattern = re.escape(old_str)
    occurrences = [
        (
            file_content.count("\\n", 0, match.start()) + 1,  # line number
            match.start(),  # start position
        )
        for match in re.finditer(pattern, file_content)
    ]
    
    # If no occurrences, try with stripped whitespace (more forgiving)
    if not occurrences:
        old_str_stripped = old_str.strip()
        new_str_stripped = new_str.strip()
        pattern = re.escape(old_str_stripped)
        occurrences = [
            (
                file_content.count("\\n", 0, match.start()) + 1,
                match.start(),
            )
            for match in re.finditer(pattern, file_content)
        ]
        if not occurrences:
            return f"Error: old_str `{old_str}` did not appear in {path}."
        # Use stripped versions
        old_str = old_str_stripped
        new_str = new_str_stripped
        pattern = re.escape(old_str)
        # Recalculate with stripped strings
        occurrences = [
            (
                file_content.count("\\n", 0, match.start()) + 1,
                match.start(),
            )
            for match in re.finditer(pattern, file_content)
        ]
    
    # Check for multiple occurrences (prevent ambiguous replacements)
    if len(occurrences) > 1:
        line_numbers = sorted(set(line for line, _ in occurrences))
        return f"Error: Multiple occurrences of old_str in lines {line_numbers}. Please ensure it is unique."
    
    # Perform replacement (we found exactly one occurrence)
    replacement_line, idx = occurrences[0]
    matched_text = old_str
    new_file_content = (
        file_content[:idx] + new_str + file_content[idx + len(matched_text):]
    )
    
    # Write back to file
    try:
        path_obj.write_text(new_file_content)
    except Exception as e:
        return f"Error: Failed to write file: {e}"
    
    # Return success message with context snippet
    context_lines = 5
    start_line = max(0, replacement_line - context_lines)
    end_line = replacement_line + context_lines + new_str.count("\\n")
    
    lines = new_file_content.split("\\n")
    snippet_lines = lines[start_line:end_line]
    snippet = "\\n".join(
        f"{{i + start_line + 1:6}}\\t{{line}}"
        for i, line in enumerate(snippet_lines)
    )
    
    return f"File {path} has been edited successfully.\\n\\n{snippet}\\n\\nReview the changes and make sure they are as expected."
"""

# ============================================================================
# 2. Grep Tool - RECOMMENDED
# ============================================================================

GREP_IMPLEMENTATION = """
def grep_tool(pattern: str, path: str = None, include: str = None) -> str:
    '''
    Search for pattern in files using ripgrep (preferred) or grep (fallback).
    
    This tool helps agents explore codebases by finding files containing
    specific patterns or keywords.
    
    Args:
        pattern: Regex pattern to search for
        path: Directory to search in (default: current directory)
        include: Glob pattern to filter files (e.g., "*.py")
    
    Returns:
        List of files containing the pattern, or error message
    '''
    import subprocess
    from pathlib import Path
    
    if path:
        search_path = Path(path).resolve()
        if not search_path.is_dir():
            return f"Error: {path} is not a valid directory"
    else:
        search_path = Path.cwd()
    
    # Try ripgrep first (faster), fallback to grep
    try:
        cmd = ["rg", "-l", "-i", pattern, str(search_path)]
        if include:
            cmd.extend(["-g", include])
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False
        )
        matches = [line for line in result.stdout.strip().split("\\n") if line]
    except FileNotFoundError:
        # Fallback to grep
        cmd = ["grep", "-r", "-l", "-I", "-i", pattern, str(search_path)]
        if include:
            cmd.extend(["--include", include])
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False
        )
        matches = [line for line in result.stdout.strip().split("\\n") if line]
    except Exception as e:
        return f"Error: Search failed: {e}"
    
    if not matches:
        include_info = f" (filtered by '{include}')" if include else ""
        return f"No files found containing pattern '{pattern}' in '{search_path}'{include_info}"
    
    # Limit to 100 files to prevent overwhelming output
    matches = matches[:100]
    truncated = len(matches) >= 100
    
    file_list = "\\n".join(matches)
    output = f"Found {len(matches)} file(s) containing pattern '{pattern}' in '{search_path}':\\n{file_list}"
    
    if truncated:
        output += "\\n\\n[Results truncated to first 100 files. Consider using a more specific pattern.]"
    
    return output
"""

# ============================================================================
# 3. Insert Tool - OPTIONAL
# ============================================================================

INSERT_IMPLEMENTATION = """
def insert_at_line(path: str, insert_line: int, new_str: str) -> str:
    '''
    Insert new_str after line insert_line in the file.
    
    This is useful for adding code at specific locations without
    overwriting the entire file.
    
    Args:
        path: Absolute path to the file
        insert_line: Line number (1-indexed) to insert after
        new_str: String to insert
    
    Returns:
        Success message or error message
    '''
    from pathlib import Path
    
    path_obj = Path(path)
    if not path_obj.exists():
        return f"Error: File {path} does not exist."
    
    if not path_obj.is_file():
        return f"Error: {path} is not a file."
    
    try:
        lines = path_obj.read_text().split("\\n")
    except Exception as e:
        return f"Error: Failed to read file: {e}"
    
    if insert_line < 0 or insert_line > len(lines):
        return f"Error: Line number {insert_line} is out of range (file has {len(lines)} lines)."
    
    # Insert after the specified line (insert_line is 1-indexed, so insert at index insert_line)
    lines.insert(insert_line, new_str)
    
    try:
        path_obj.write_text("\\n".join(lines))
    except Exception as e:
        return f"Error: Failed to write file: {e}"
    
    return f"Successfully inserted text after line {insert_line} in {path}."
"""

# ============================================================================
# Prompt Template for Self-Improvement
# ============================================================================

OPENHANDS_TOOLS_REFERENCE_PROMPT = """
## Reference: OpenHands SDK Tool Implementations

The following are high-quality tool implementations from OpenHands SDK (a production-grade agent framework) that you can reference when analyzing and improving the DGM agent's tools.

### 1. String Replace Tool (str_replace) - HIGHLY RECOMMENDED

This tool allows replacing a specific string in a file, which is more efficient than overwriting the entire file. This is especially useful for large files where only a small change is needed.

**Key Features**:
- Replaces `old_str` with `new_str` in a file
- Ensures `old_str` appears exactly once (prevents ambiguous replacements)
- Handles whitespace variations automatically (strips whitespace if exact match fails)
- Returns a snippet of the edited section for verification
- More efficient than full file overwrite

**Implementation**:
```python
{str_replace_impl}
```

**Benefits for DGM**:
- Current `edit.py` only supports full file overwrite (`edit` command)
- Adding `str_replace` would allow more efficient incremental edits
- Reduces risk of accidental changes to unrelated parts of the file
- Better for large files

### 2. Grep Tool - RECOMMENDED

A code search tool that finds files containing a specific pattern. This helps agents explore codebases more effectively.

**Key Features**:
- Uses ripgrep (fast) with fallback to grep
- Supports file pattern filtering (e.g., `*.py`)
- Limits results to prevent overwhelming output
- Case-insensitive search

**Implementation**:
```python
{grep_impl}
```

**Benefits for DGM**:
- Better code exploration capabilities
- Can help agent understand codebase structure
- Useful for finding where functions/classes are used

### 3. Insert Tool - OPTIONAL

Insert text at a specific line number. Useful for adding code at specific locations.

**Key Features**:
- Inserts text after a specified line number
- Simple and straightforward
- Validates line number range

**Implementation**:
```python
{insert_impl}
```

**Benefits for DGM**:
- Useful for adding code at specific locations
- Complements `str_replace` functionality
- More precise than full file overwrite

## Task

When analyzing the current agent implementation, consider:

1. **Could the current `edit.py` tool benefit from `str_replace` functionality?**
   - Current `edit` command overwrites entire file
   - `str_replace` would allow incremental edits
   - Could add `str_replace` as a new command to `edit.py`

2. **Would a `grep` tool help the agent explore codebases more effectively?**
   - Current tools don't have dedicated code search
   - Agent relies on bash commands for searching
   - A dedicated tool might be more reliable

3. **How can these implementations be adapted to fit DGM's architecture?**
   - DGM uses simple `tool_info()` and `tool_function()` interface
   - Tools should be stateless (no persistent sessions)
   - Keep implementations simple and aligned with DGM's minimal design

## What You Can Propose

You can propose:
- **Adding new commands** to existing tools (e.g., adding "str_replace" command to `edit.py`)
- **Creating new tools** based on these references (e.g., creating a new `grep.py` tool)
- **Improving existing tools** by learning from these implementations (e.g., better error handling)

Remember to:
- Keep implementations simple and aligned with DGM's architecture
- Maintain the `tool_info()` and `tool_function()` interface
- Ensure tools work in a stateless manner (no persistent sessions)
- Follow the pattern of existing tools in `tools/` directory
"""


def get_openhands_tools_reference_prompt() -> str:
    """
    Get the formatted prompt with tool implementations.
    
    Returns:
        Formatted prompt string ready to be inserted into self-improvement prompts
    """
    return OPENHANDS_TOOLS_REFERENCE_PROMPT.format(
        str_replace_impl=STR_REPLACE_IMPLEMENTATION,
        grep_impl=GREP_IMPLEMENTATION,
        insert_impl=INSERT_IMPLEMENTATION,
    )

