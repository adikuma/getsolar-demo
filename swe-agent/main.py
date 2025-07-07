import os
import sys
from pathlib import Path
from uuid import uuid4

import click
from rich.console import Console
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import clear
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter

# imports with fallbacks
try:
    from agent.core import SWEAgent
except ImportError:
    class SWEAgent:
        def process_sync(self, text, thread_id):
            return f"mock response to: {text}"
        def get_conversation_history(self, thread_id):
            return []

try:
    from utils.config import Config
except ImportError:
    class Config:
        def __init__(self):
            self.debug_mode = False

console = Console()

class SWEAgentCLI:
    def __init__(self):
        self.config = Config()
        self.agent = SWEAgent(self.config)
        self.history = FileHistory(str(Path.home() / ".swe_agent_history"))
        self.session_count = 0
        
        # persistent thread for conversation memory
        self.thread_id = str(uuid4())
        
        # command completion
        self.completer = WordCompleter([
            "/help", "/clear", "/reset", "/status", "/debug", "/quit", "/test"
        ])

    def _banner(self):
        console.print("swe agent v1.0")
        console.print("type /help for commands\n")

    def _prompt_text(self) -> HTML:
        return HTML(f"<ansiblue>swe</ansiblue> > ")

    def _handle_command(self, user_input: str) -> str | None:
        if not user_input.startswith("/"):
            return None

        cmd = user_input[1:].lower().strip()
        
        if cmd == "help":
            return self._help()
        elif cmd == "clear":
            clear()
            self._banner()
            return "screen cleared"
        elif cmd == "reset":
            old_thread = self.thread_id[:8]
            self.thread_id = str(uuid4())
            return f"conversation reset (was: {old_thread}...)"
        elif cmd == "status":
            return self._status()
        elif cmd == "debug":
            self.config.debug_mode = not self.config.debug_mode
            status = "on" if self.config.debug_mode else "off"
            return f"debug mode {status}"
        elif cmd in ["quit", "exit"]:
            console.print("goodbye")
            sys.exit(0)
        elif cmd == "test":
            return self._test()
        else:
            return f"unknown command: /{cmd}"

    def _help(self) -> str:
        console.print("commands:")
        console.print("  /help   - show this help")
        console.print("  /clear  - clear screen")
        console.print("  /reset  - start new conversation")
        console.print("  /status - show session info")
        console.print("  /debug  - toggle debug mode")
        console.print("  /test   - run system test")
        console.print("  /quit   - exit")
        return ""

    def _status(self) -> str:
        try:
            history = self.agent.get_conversation_history(self.thread_id)
            msg_count = len([m for m in history if hasattr(m, 'content')])
            return f"session: {self.session_count}\nthread: {self.thread_id[:8]}...\nmessages: {msg_count}"
        except Exception as e:
            return f"status error: {str(e)}"

    def _test(self) -> str:
        results = []
        
        # check api keys
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        
        results.append(f"anthropic key: {'ok' if anthropic_key else 'missing'}")
        results.append(f"tavily key: {'ok' if tavily_key else 'missing'}")
        
        # test agent
        try:
            response = self.agent.process_sync("hello", self.thread_id)
            if response and len(response) > 0:
                results.append("agent: ok")
            else:
                results.append("agent: empty response")
        except Exception as e:
            results.append(f"agent: error - {str(e)}")
        
        return "\n".join(results)

    def _process_input(self, user_input: str) -> str:
        try:
            console.print("thinking...")
            result = self.agent.process_sync(user_input, self.thread_id)
            return result or "no response received"
        except Exception as e:
            if self.config.debug_mode:
                console.print_exception()
            return f"error: {str(e)}"

    def _format_response(self, response: str):
        if not response.strip():
            console.print("[dim]empty response[/dim]")
            return
        
        console.print(f"\n{response}\n")

    def run(self):
        self._banner()
        
        while True:
            try:
                user_input = prompt(
                    self._prompt_text(),
                    history=self.history,
                    auto_suggest=AutoSuggestFromHistory(),
                    completer=self.completer,
                ).strip()

                if not user_input:
                    continue

                # handle commands
                cmd_response = self._handle_command(user_input)
                if cmd_response is not None:
                    if cmd_response:
                        console.print(cmd_response)
                    continue

                # process with agent
                response = self._process_input(user_input)
                self._format_response(response)
                self.session_count += 1

            except KeyboardInterrupt:
                console.print("\nuse /quit to exit")
                continue
            except EOFError:
                console.print("\ngoodbye")
                break
            except Exception as e:
                if self.config.debug_mode:
                    console.print_exception()
                else:
                    console.print(f"error: {str(e)}")

@click.command()
@click.option("--debug", is_flag=True, help="enable debug mode")
def main(debug: bool):
    cli = SWEAgentCLI()
    if debug:
        cli.config.debug_mode = True
    
    try:
        cli.run()
    except KeyboardInterrupt:
        console.print("\ninterrupted")
    except Exception as e:
        console.print(f"fatal: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()