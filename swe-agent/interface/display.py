import re
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.columns import Columns
from rich.text import Text


class DisplayManager:
    """manages rich output formatting and display elements"""
    
    def __init__(self, console: Console):
        self.console = console
        
    def render_markdown_with_code(self, content: str) -> None:
        """render markdown content with syntax highlighting for code blocks"""
        # split content by code blocks
        parts = re.split(r'```(\w+)?\n(.*?)\n```', content, flags=re.DOTALL)
        
        for i, part in enumerate(parts):
            if i % 3 == 0:  # regular markdown
                if part.strip():
                    self.console.print(Markdown(part.strip()))
            elif i % 3 == 1:  # language identifier
                continue
            else:  # code content
                language = parts[i-1] if parts[i-1] else "text"
                if part.strip():
                    syntax = Syntax(part.strip(), language, theme="monokai", line_numbers=True)
                    self.console.print(Panel(syntax, title=f"code ({language})", border_style="blue"))
                    
    def render_agent_response(self, response: str, title: str = "reasoning agent") -> None:
        """render formatted agent response with appropriate styling"""
        if response.startswith('❌') or 'error' in response.lower():
            self.console.print(Panel(
                response, 
                title=f"[red]{title}[/red]", 
                border_style="red",
                padding=(1, 1)
            ))
        elif response.startswith('✅') or 'success' in response.lower():
            self.console.print(Panel(
                response, 
                title=f"[green]{title}[/green]", 
                border_style="green",
                padding=(1, 1)
            ))
        else:
            # check for structured content
            if self._contains_structured_content(response):
                self.render_markdown_with_code(response)
            else:
                self.console.print(Panel(
                    response, 
                    title=title, 
                    border_style="blue",
                    padding=(1, 1)
                ))
                
    def _contains_structured_content(self, text: str) -> bool:
        """check if text contains markdown-like structured content"""
        indicators = ['```', '##', '**', '1.', '2.', '- ', '* ']
        return any(indicator in text for indicator in indicators)
        
    def render_tool_execution(self, tool_name: str, status: str, output: str) -> None:
        """render tool execution results with status indicators"""
        status_colors = {
            'success': 'green',
            'error': 'red', 
            'warning': 'yellow',
            'info': 'blue'
        }
        
        color = status_colors.get(status, 'white')
        
        panel = Panel(
            output,
            title=f"[{color}]{tool_name}[/{color}] [{color}]({status})[/{color}]",
            border_style=color,
            padding=(0, 1)
        )
        
        self.console.print(panel)
        
    def render_file_analysis(self, analysis_data: Dict[str, Any]) -> None:
        """render structured file analysis results"""
        # create main info table
        info_table = Table(title="file analysis", show_header=False)
        info_table.add_column("property", style="cyan", no_wrap=True)
        info_table.add_column("value", style="white")
        
        basic_info = analysis_data.get('basic_info', {})
        for key, value in basic_info.items():
            info_table.add_row(key, str(value))
            
        self.console.print(info_table)
        
        # render code metrics if available
        metrics = analysis_data.get('code_metrics', {})
        if metrics:
            metrics_table = Table(title="code metrics")
            metrics_table.add_column("metric", style="cyan")
            metrics_table.add_column("count", style="yellow", justify="right")
            
            for metric, count in metrics.items():
                metrics_table.add_row(metric, str(count))
                
            self.console.print(metrics_table)
            
        # render code preview
        preview = analysis_data.get('preview', '')
        if preview:
            language = analysis_data.get('language', 'text')
            syntax = Syntax(preview, language, theme="monokai", line_numbers=True)
            self.console.print(Panel(syntax, title="code preview", border_style="green"))
            
    def render_search_results(self, results: List[Dict[str, str]]) -> None:
        """render web search results in a structured format"""
        for i, result in enumerate(results, 1):
            title = result.get('title', 'no title')
            url = result.get('url', 'no url')
            content = result.get('content', 'no content')
            
            # create result panel
            result_content = f"""
            [bold blue]{title}[/bold blue]
            [dim]{url}[/dim]

            {content}
            """
            
            self.console.print(Panel(
                result_content,
                title=f"result {i}",
                border_style="cyan",
                padding=(1, 1)
            ))
            
    def render_project_structure(self, structure_data: Dict[str, Any]) -> None:
        """render project structure analysis"""
        # file distribution table
        dist_table = Table(title="file type distribution")
        dist_table.add_column("extension", style="cyan")
        dist_table.add_column("files", style="yellow", justify="right")
        dist_table.add_column("size", style="green", justify="right")
        
        file_types = structure_data.get('file_types', {})
        for ext, info in sorted(file_types.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
            ext_display = ext if ext else 'no extension'
            size_mb = info['size'] / 1024 / 1024
            dist_table.add_row(ext_display, str(info['count']), f"{size_mb:.1f} mb")
            
        self.console.print(dist_table)
        
        # project characteristics
        characteristics = structure_data.get('characteristics', {})
        if characteristics:
            char_text = []
            for feature, present in characteristics.items():
                icon = "✅" if present else "❌"
                char_text.append(f"{icon} {feature}")
                
            self.console.print(Panel(
                "\n".join(char_text),
                title="project characteristics",
                border_style="blue"
            ))
            
    def render_progress_spinner(self, description: str = "processing..."):
        """create and return a progress spinner context manager"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        )
        
    def render_error(self, error_message: str, title: str = "error") -> None:
        """render error messages with appropriate styling"""
        self.console.print(Panel(
            f"[red]{error_message}[/red]",
            title=f"[red]{title}[/red]",
            border_style="red",
            padding=(1, 1)
        ))
        
    def render_success(self, message: str, title: str = "success") -> None:
        """render success messages with appropriate styling"""
        self.console.print(Panel(
            f"[green]{message}[/green]",
            title=f"[green]{title}[/green]",
            border_style="green",
            padding=(1, 1)
        ))
        
    def render_warning(self, message: str, title: str = "warning") -> None:
        """render warning messages with appropriate styling"""
        self.console.print(Panel(
            f"[yellow]{message}[/yellow]",
            title=f"[yellow]{title}[/yellow]",
            border_style="yellow",
            padding=(1, 1)
        ))
        
    def render_info(self, message: str, title: str = "info") -> None:
        """render info messages with appropriate styling"""
        self.console.print(Panel(
            f"[blue]{message}[/blue]",
            title=f"[blue]{title}[/blue]",
            border_style="blue",
            padding=(1, 1)
        ))
        
    def render_comparison_table(self, comparisons: List[Dict[str, str]], title: str = "comparison") -> None:
        """render comparison data in table format"""
        if not comparisons:
            return
            
        table = Table(title=title)
        
        # add columns based on first item keys
        for key in comparisons[0].keys():
            table.add_column(key, style="cyan")
            
        # add rows
        for item in comparisons:
            table.add_row(*[str(value) for value in item.values()])
            
        self.console.print(table)
        
    def render_tree_structure(self, tree_data: Dict[str, Any], title: str = "structure") -> None:
        """render hierarchical data as a tree"""
        tree = Tree(title)
        
        def add_items(parent_node, items):
            for key, value in items.items():
                if isinstance(value, dict):
                    node = parent_node.add(f"[bold cyan]{key}[/bold cyan]")
                    add_items(node, value)
                else:
                    parent_node.add(f"{key}: [yellow]{value}[/yellow]")
                    
        add_items(tree, tree_data)
        self.console.print(tree)
        
    def render_columns(self, content_list: List[str], title: str = "") -> None:
        """render content in columns for better space utilization"""
        if title:
            self.console.print(f"\n[bold]{title}[/bold]")
            
        columns = Columns(content_list, equal=True, expand=True)
        self.console.print(columns)
        
    def clear_screen(self) -> None:
        """clear the console screen"""
        self.console.clear()
        
    def print_separator(self, char: str = "=", length: int = 50) -> None:
        """print a visual separator"""
        self.console.print(char * length, style="dim")