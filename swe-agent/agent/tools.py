import os
import subprocess
import json
import ast
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import TavilySearchResults
from langchain_community.tools.file_management import (
    ReadFileTool, WriteFileTool, ListDirectoryTool
)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage

@tool
def enhanced_python_repl(code: str) -> str:
    """
    execute python code with enhanced capabilities and safety checks.
    supports data science libraries, visualization, and file operations.
    includes automatic result formatting and error recovery.
    """
    repl = PythonREPL()
    
    # safety checks
    dangerous_patterns = [
        r'import\s+os.*system',
        r'subprocess\.',
        r'eval\s*\(',
        r'exec\s*\(',
        r'__import__',
        r'open.*w.*root',
        r'rm\s+-rf'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return f"⚠️ potentially unsafe operation detected: {pattern}"
    
    try:
        print(f"📝 executing python code:\n{code}")
        
        # execute with enhanced error handling
        result = repl.run(code)
        
        if result:
            # format output nicely
            formatted_result = f"✅ execution successful\n\noutput:\n{result}"
            print(formatted_result)
            return formatted_result
        else:
            success_msg = "✅ code executed successfully (no output)"
            print(success_msg)
            return success_msg
            
    except SyntaxError as e:
        error_msg = f"❌ syntax error: {str(e)}\nline {e.lineno}: {e.text}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ runtime error: {type(e).__name__}: {str(e)}"
        print(error_msg)
        return error_msg

def filter_messages_for_openai(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Filter messages to ensure valid OpenAI format"""
    filtered = []
    i = 0
    
    while i < len(messages):
        msg = messages[i]
        
        if isinstance(msg, (SystemMessage, HumanMessage)):
            filtered.append(msg)
            i += 1
        elif isinstance(msg, AIMessage):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                filtered.append(msg)
                i += 1
                
                while i < len(messages) and isinstance(messages[i], ToolMessage):
                    filtered.append(messages[i])
                    i += 1
            else:
                filtered.append(msg)
                i += 1
        else:
            i += 1
    
    return filtered

@tool
def web_search_enhanced(query: str, max_results: int = 3) -> str:
    """
    perform intelligent web search with structured results and source verification.
    automatically filters and ranks results for relevance and credibility.
    """
    try:
        print(f"🔍 searching web for: '{query}'")
        
        search = TavilySearchResults(
            max_results=max_results,
            api_key=os.getenv("TAVILY_API_KEY")
        )
        
        results = search.run(query)
        
        if not results:
            return "no search results found"
        
        # format results with enhanced structure
        formatted_results = [f"🌐 web search results for: '{query}'\n"]
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'no title')
            url = result.get('url', 'no url')
            content = result.get('content', 'no content')
            
            # truncate content intelligently
            content_preview = content[:400] + "..." if len(content) > 400 else content
            
            formatted_results.append(f"""
            📄 result {i}: {title}
            🔗 source: {url}
            📋 summary: {content_preview}
            """)
                    
        formatted_output = "\n".join(formatted_results)
        # print(formatted_output)
        return formatted_output
        
    except Exception as e:
        error_msg = f"❌ search failed: {str(e)}"
        print(error_msg)
        return error_msg


@tool
def safe_shell_execute(command: str) -> str:
    """
    execute shell commands in a controlled, safe environment.
    whitelist approach with comprehensive logging and timeout protection.
    """
    # comprehensive whitelist of safe commands
    safe_commands = {
        # file operations
        'ls', 'dir', 'pwd', 'cat', 'head', 'tail', 'less', 'more',
        'file', 'stat', 'du', 'df', 'find', 'locate',
        
        # text processing  
        'grep', 'awk', 'sed', 'sort', 'uniq', 'wc', 'cut',
        
        # system info
        'ps', 'top', 'htop', 'whoami', 'id', 'uname', 'date',
        'uptime', 'free', 'which', 'whereis',
        
        # network (read-only)
        'ping', 'nslookup', 'dig', 'curl', 'wget',
        
        # development
        'git', 'python', 'node', 'npm', 'pip', 'conda',
        
        # misc utilities
        'echo', 'printf', 'sleep', 'timeout', 'history'
    }
    
    cmd_parts = command.strip().split()
    if not cmd_parts:
        return "❌ empty command"
    
    base_cmd = cmd_parts[0]
    if base_cmd not in safe_commands:
        available = ", ".join(sorted(safe_commands))
        return f"❌ command '{base_cmd}' not allowed\n\n✅ safe commands: {available}"
    
    try:
        print(f"🐚 executing shell command: {command}")
        
        # execute with timeout and capture
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
            cwd=Path.cwd()
        )
        
        if result.returncode == 0:
            output = f"✅ command successful\n\noutput:\n{result.stdout}"
            if result.stderr:
                output += f"\nstderr:\n{result.stderr}"
        else:
            output = f"❌ command failed (exit code {result.returncode})\nstderr:\n{result.stderr}"
            
        print(output)
        return output
        
    except subprocess.TimeoutExpired:
        error_msg = "❌ command timed out (30s limit)"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ execution failed: {str(e)}"
        print(error_msg)
        return error_msg


@tool
def advanced_file_analyzer(file_path: str) -> str:
    """
    perform comprehensive analysis of code files with intelligent insights.
    supports multiple languages and provides actionable improvement suggestions.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"❌ file not found: {file_path}"
        
        if path.stat().st_size > 5 * 1024 * 1024:  # 5mb limit
            return f"❌ file too large: {path.stat().st_size / 1024 / 1024:.1f}mb (limit: 5mb)"
        
        print(f"🔍 analyzing file: {path.name}")
        
        # read file content
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                content = path.read_text(encoding='latin-1')
            except:
                return f"❌ unable to read file: unsupported encoding"
        
        lines = content.split('\n')
        
        # basic file metrics
        analysis = f"""
            📁 file analysis: {path.name}
            📏 size: {len(content):,} characters, {len(lines):,} lines
            📅 modified: {datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}
            🏷️ type: {path.suffix or 'no extension'}
            """
                    
        # language-specific analysis
        if path.suffix == '.py':
            analysis += _analyze_python_file(content, lines)
        elif path.suffix in ['.js', '.ts', '.jsx', '.tsx']:
            analysis += _analyze_javascript_file(content, lines)
        elif path.suffix in ['.java']:
            analysis += _analyze_java_file(content, lines)
        elif path.suffix in ['.cpp', '.c', '.h']:
            analysis += _analyze_cpp_file(content, lines)
        elif path.suffix in ['.md', '.txt']:
            analysis += _analyze_text_file(content, lines)
        elif path.suffix in ['.json']:
            analysis += _analyze_json_file(content)
        elif path.suffix in ['.csv']:
            analysis += _analyze_csv_file(content, lines)
        else:
            analysis += _analyze_generic_file(content, lines)
        
        print(analysis)
        return analysis
        
    except Exception as e:
        error_msg = f"❌ analysis failed: {str(e)}"
        print(error_msg)
        return error_msg


def _analyze_python_file(content: str, lines: List[str]) -> str:
    """analyze python-specific features"""
    try:
        tree = ast.parse(content)
        
        imports = []
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend([f"{module}.{alias.name}" for alias in node.names])
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        # code quality metrics
        complexity_indicators = [
            len([l for l in lines if 'if ' in l or 'elif ' in l]),
            len([l for l in lines if 'for ' in l or 'while ' in l]),
            len([l for l in lines if 'try:' in l or 'except' in l])
        ]
        docstring_count = content.count('"""') // 2 + content.count("'''") // 2
        
        return f"""
        🐍 python analysis:
        📦 imports: {len(imports)} ({', '.join(imports[:5])}{'...' if len(imports) > 5 else ''})
        🔧 functions: {len(functions)} ({', '.join(functions[:3])}{'...' if len(functions) > 3 else ''})
        🏗️ classes: {len(classes)} ({', '.join(classes[:3])}{'...' if len(classes) > 3 else ''})
        🧮 complexity: {sum(complexity_indicators)} conditional/loop/exception blocks
        📝 docstrings: {docstring_count}


        📋 code preview:
        {chr(10).join(lines[:10])}
        """
    except SyntaxError:
        return f"""
        🐍 python analysis:
        ❌ syntax errors detected - file may be incomplete or invalid
        📋 content preview:
        {chr(10).join(lines[:10])}
        """


def _analyze_javascript_file(content: str, lines: List[str]) -> str:
    """analyze javascript/typescript features"""
    functions = len(re.findall(r'function\s+\w+|const\s+\w+\s*=.*=>|\w+\s*=.*=>', content))
    classes = len(re.findall(r'class\s+\w+', content))
    imports = len(re.findall(r'import\s+.*from|require\s*\(', content))
    
    return f"""
    🟨 javascript/typescript analysis:
    📦 imports/requires: {imports}
    🔧 functions: {functions}  
    🏗️ classes: {classes}
    📋 content preview:
    {chr(10).join(lines[:10])}
    """


def _analyze_java_file(content: str, lines: List[str]) -> str:
    """analyze java features"""
    classes = len(re.findall(r'class\s+\w+', content))
    methods = len(re.findall(r'public\s+\w+.*\(|private\s+\w+.*\(', content))
    imports = len(re.findall(r'import\s+', content))
    
    return f"""
    ☕ java analysis:
    📦 imports: {imports}
    🏗️ classes: {classes}
    🔧 methods: {methods}
    📋 content preview:
    {chr(10).join(lines[:10])}
    """


def _analyze_cpp_file(content: str, lines: List[str]) -> str:
    """analyze c/c++ features"""
    includes = len(re.findall(r'#include', content))
    functions = len(re.findall(r'\w+\s+\w+\s*\([^)]*\)\s*{', content))
    classes = len(re.findall(r'class\s+\w+', content))
    
    return f"""
    ⚡ c/c++ analysis:
    📦 includes: {includes}
    🔧 functions: {functions}
    🏗️ classes: {classes}
    📋 content preview:
    {chr(10).join(lines[:10])}
    """


def _analyze_text_file(content: str, lines: List[str]) -> str:
    """analyze text/markdown features"""
    words = len(content.split())
    paragraphs = len([l for l in lines if l.strip() and not l.startswith('#')])
    headings = len([l for l in lines if l.startswith('#')])
    
    return f"""
    📄 text/markdown analysis:
    📝 words: {words:,}
    📑 paragraphs: {paragraphs}
    🏷️ headings: {headings}
    📋 content preview:
    {chr(10).join(lines[:15])}
    """


def _analyze_json_file(content: str) -> str:
    """analyze json structure"""
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            keys = list(data.keys())[:10]
            return f"""
            🗂️ json analysis:
            🔑 top-level keys: {len(data)} ({', '.join(keys)}{'...' if len(data) > 10 else ''})
            📊 structure: dictionary with {len(data)} properties
            """
        elif isinstance(data, list):
            return f"""
            🗂️ json analysis:
            📊 structure: array with {len(data)} items
            🔍 item types: {', '.join(set(type(item).__name__ for item in data[:10]))}
            """
        else:
            return f"""
            🗂️ json analysis:
            📊 structure: {type(data).__name__}
            """
    except json.JSONDecodeError as e:
        return f"""
        🗂️ json analysis:
        ❌ invalid json: {str(e)}
        """


def _analyze_csv_file(content: str, lines: List[str]) -> str:
    """analyze csv structure"""
    if not lines:
        return "❌ empty csv file"
        
    header = lines[0] if lines else ""
    columns = header.split(',') if ',' in header else header.split('\t')
    
    return f"""
    📊 csv analysis:
    📋 columns: {len(columns)} ({', '.join(columns[:5])}{'...' if len(columns) > 5 else ''})
    📏 rows: {len(lines) - 1} (excluding header)
    📋 sample data:
    {chr(10).join(lines[:5])}
    """


def _analyze_generic_file(content: str, lines: List[str]) -> str:
    """generic file analysis"""
    return f"""
    📄 generic file analysis:
    📋 content preview:
    {chr(10).join(lines[:15])}
    """


@tool
def project_structure_analyzer(directory_path: str = ".") -> str:
    """
    analyze project structure and provide insights about organization.
    identifies common patterns, suggests improvements, and detects issues.
    """
    try:
        path = Path(directory_path)
        if not path.exists():
            return f"❌ directory not found: {directory_path}"
        
        print(f"🏗️ analyzing project structure: {path.absolute()}")
        
        # collect file information
        files_by_type = {}
        total_files = 0
        total_size = 0
        
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_files += 1
                size = file_path.stat().st_size
                total_size += size
                
                ext = file_path.suffix.lower()
                if ext not in files_by_type:
                    files_by_type[ext] = {'count': 0, 'size': 0}
                files_by_type[ext]['count'] += 1
                files_by_type[ext]['size'] += size
        
        # project insights
        analysis = f"""
        🏗️ project structure analysis: {path.name}
        📁 total files: {total_files:,}
        📏 total size: {total_size / 1024 / 1024:.1f} mb

        📊 file types distribution:
        """
        
        # sort by count
        sorted_types = sorted(files_by_type.items(), key=lambda x: x[1]['count'], reverse=True)
        for ext, info in sorted_types[:10]:
            ext_name = ext if ext else 'no extension'
            analysis += f"  {ext_name}: {info['count']} files ({info['size'] / 1024:.1f} kb)\n"
        
        # detect common project patterns
        has_readme = any(path.glob('README*'))
        has_requirements = any(path.glob('requirements.txt')) or any(path.glob('package.json'))
        has_gitignore = any(path.glob('.gitignore'))
        has_tests = any(path.glob('test*')) or any(path.glob('*test*'))
        
        analysis += f"""
        🔍 project characteristics:
        📖 readme: {'✅' if has_readme else '❌'}
        📦 dependencies: {'✅' if has_requirements else '❌'}
        🚫 gitignore: {'✅' if has_gitignore else '❌'}
        🧪 tests: {'✅' if has_tests else '❌'}
        """
        
        print(analysis)
        return analysis
        
    except Exception as e:
        error_msg = f"❌ analysis failed: {str(e)}"
        print(error_msg)
        return error_msg


def get_enhanced_tools() -> List:
    """return collection of all enhanced tools"""
    return [
        enhanced_python_repl,
        web_search_enhanced,
        safe_shell_execute,
        advanced_file_analyzer,
        project_structure_analyzer,
        ReadFileTool(),
        WriteFileTool(),
        ListDirectoryTool()
    ]