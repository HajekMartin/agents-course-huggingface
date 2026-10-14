"""Use smolagents' streaming chat with a UI that needs no remote image assets."""

import gradio as gr
from smolagents import GradioUI as BaseGradioUI


class GradioUI(BaseGradioUI):
    def __init__(self, agent):
        # Failed actions from an earlier message must not become the next task's examples.
        super().__init__(agent, reset_agent_memory=True)

    def create_app(self):
        """Keep the framework's agent trace and chat handling, with a local interface."""
        return gr.ChatInterface(
            fn=self._stream_response,
            title="Local Ollama Agent",
            description="Roll dice, check your computer's timezone, or get the time in an IANA timezone.",
            examples=[
                "Roll a random number of dice between 2 and 10 times. Show every result and the total.",
                "Roll exactly 5 times and show the results.",
                "What time is it in Europe/Prague?",
                "What is my current timezone?",
            ],
            cache_examples=False,
            save_history=True,
        )
