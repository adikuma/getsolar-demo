import webbrowser
from typing import List, Dict, Any
from langchain_core.tools import tool

URL = "https://calculator.getsolar.ai/en-SG/"

@tool
def open_getsolar_calculator() -> str:
    """
    description:
        open the user's default web browser to the GetSolar solar calculator.
        once the user has finished interacting, they press ENTER in this console
        to return control.

    args:
        dummy_input (str): ignored placeholder (langchain tools require one arg)

    return:
        str: a message indicating the user has completed interaction
    """
    opened = webbrowser.open(URL)
    if not opened:
        # fallback to printing the URL
        print(f" please manually open your browser to: {URL}")

    input(" when you’ve finished using the solar calculator, press ENTER here…")

    return "The user has completed interacting with the GetSolar calculator"

if __name__ == "__main__":
    print(" opening GetSolar calculator in your browser…")
    result = open_getsolar_calculator("run")
    print(" tool returned:", result)