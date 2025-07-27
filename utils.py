import streamlit as st
import re

def show_error(msg: str):
    st.error(f"Error: {msg}")

def remove_think_content(text: str) -> str:
    """
    Removes all text enclosed within <think>...</think> tags.

    Args:
        text (str): The original chunk text.

    Returns:
        str: Text with all <think>...</think> blocks removed.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()