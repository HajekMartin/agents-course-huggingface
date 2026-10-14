"""Verify local tools and the CodeAgent action/observation loop without an LLM."""

import unittest
from datetime import datetime
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import requests
from smolagents.models import ChatMessage, MessageRole

from app import (
    check_ollama,
    create_agent,
    get_current_time_in_timezone,
    get_local_timezone,
    roll_dice,
    validate_ollama_host,
)


class DiceTests(unittest.TestCase):
    def test_random_count_and_results(self):
        with patch("app.random.randint", side_effect=[3, 2, 6, 1]) as randint:
            result = roll_dice(2, 10)
        self.assertEqual(result, {"count": 3, "rolls": [2, 6, 1], "total": 9})
        self.assertEqual(randint.call_args_list[0].args, (2, 10))
        self.assertTrue(all(call.args == (1, 6) for call in randint.call_args_list[1:]))

    def test_inclusive_count_boundaries(self):
        for count in (1, 6, 1000):
            with self.subTest(count=count):
                result = roll_dice(count, count)
                self.assertEqual(result["count"], count)
                self.assertEqual(len(result["rolls"]), count)
                self.assertTrue(all(1 <= value <= 6 for value in result["rolls"]))
                self.assertEqual(result["total"], sum(result["rolls"]))

    def test_invalid_bounds_do_not_generate_results(self):
        for bounds in ((0, 6), (3, 2), (1, 1001), (1.5, 6), (True, 6)):
            with self.subTest(bounds=bounds), patch("app.random.randint") as randint:
                with self.assertRaises(ValueError):
                    roll_dice.forward(*bounds)
                randint.assert_not_called()


class TimeTests(unittest.TestCase):
    def test_local_timezone_and_offset_follow_system_zone_and_season(self):
        cases = [
            ("Europe/Prague", 1, "+01:00", "CET"),
            ("Europe/Prague", 7, "+02:00", "CEST"),
            ("America/New_York", 1, "-05:00", "EST"),
            ("Asia/Kathmandu", 1, "+05:45", "+0545"),
        ]
        for name, month, offset, abbreviation in cases:
            with self.subTest(zone=name, month=month):
                zone = ZoneInfo(name)
                now = datetime(2026, month, 15, 12, 30, tzinfo=zone)
                with patch("app.get_localzone", return_value=zone), patch("app.datetime") as clock:
                    clock.now.return_value = now
                    result = get_local_timezone()
                clock.now.assert_called_once_with(zone)
                self.assertEqual(result["timezone"], name)
                self.assertEqual(result["utc_offset"], offset)
                self.assertEqual(result["abbreviation"], abbreviation)
                self.assertEqual(result["local_time"], now.isoformat(timespec="seconds"))
                self.assertIn("this computer", result["source"])

    def test_local_timezone_detection_error_is_reported_without_guessing(self):
        with patch("app.get_localzone", side_effect=ValueError("Conflicting timezone settings")):
            result = get_local_timezone()
        self.assertIn("Conflicting timezone settings", result["error"])
        self.assertNotIn("timezone", result)

    def test_utc_time_matches_clock(self):
        with patch("app.datetime") as clock:
            clock.now.return_value = datetime(2026, 1, 15, 12, 30, tzinfo=ZoneInfo("UTC"))
            result = get_current_time_in_timezone("UTC")
        self.assertIn("2026-01-15 12:30:00 UTC (+0000)", result)

    def test_invalid_timezone(self):
        self.assertIn("Unknown timezone", get_current_time_in_timezone("Invalid/Timezone"))


class OllamaTests(unittest.TestCase):
    def test_only_local_root_urls_are_accepted(self):
        for host in ("http://localhost:11434/", "http://127.0.0.1:11434", "http://[::1]:11434"):
            self.assertEqual(validate_ollama_host(host), host.rstrip("/"))
        for host in ("https://example.com", "localhost:11434", "http://localhost:11434/v1"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                validate_ollama_host(host)

    def test_installed_model_and_latest_alias(self):
        response = Mock()
        response.json.return_value = {"models": [{"name": "qwen2:7b"}, {"name": "test:latest"}]}
        with patch("app.requests.get", return_value=response):
            check_ollama("http://localhost:11434", "qwen2:7b")
            check_ollama("http://localhost:11434", "test")
            with self.assertRaisesRegex(RuntimeError, "ollama pull missing"):
                check_ollama("http://localhost:11434", "missing")

    def test_unreachable_server_has_actionable_error(self):
        with patch("app.requests.get", side_effect=requests.ConnectionError):
            with self.assertRaisesRegex(RuntimeError, "ollama serve"):
                check_ollama("http://localhost:11434", "qwen2:7b")


class AgentTests(unittest.TestCase):
    def test_local_timezone_task_executes_tool_and_finishes(self):
        agent = create_agent("qwen2:7b", "http://localhost:11434")
        agent.logger.level = -1
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Thought: Read this computer's timezone.\n<code>\nresult = get_local_timezone()\nfinal_answer(result)\n</code>",
        )
        zone = ZoneInfo("Europe/Prague")
        with patch.object(agent.model, "generate", return_value=message) as generate:
            with patch("app.get_localzone", return_value=zone) as detect, patch("app.datetime") as clock:
                clock.now.return_value = datetime(2026, 7, 15, 12, 30, tzinfo=zone)
                result = agent.run("What is my current timezone?")
        detect.assert_called_once()
        generate.assert_called_once()
        self.assertEqual(result["timezone"], "Europe/Prague")
        self.assertEqual(result["utc_offset"], "+02:00")
        self.assertIsNone(agent.memory.steps[-1].error)

    def test_greeting_finishes_without_other_tools(self):
        agent = create_agent("qwen2:7b", "http://localhost:11434")
        agent.logger.level = -1
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content='Thought: Answer the greeting.\n<code>\nfinal_answer("Ahoj!")\n</code>',
        )
        with patch.object(agent.model, "generate", return_value=message) as generate:
            with patch("app.random.randint") as randint:
                self.assertEqual(agent.run("ahoj"), "Ahoj!")
        generate.assert_called_once()
        randint.assert_not_called()
        self.assertIsNone(agent.memory.steps[-1].error)

    def test_ui_does_not_reuse_failed_steps_from_previous_task(self):
        from Gradio_UI import GradioUI

        agent = create_agent("qwen2:7b", "http://localhost:11434")
        agent.logger.level = -1
        ui = GradioUI(agent)
        messages = [
            ChatMessage(role=MessageRole.ASSISTANT, content="Final Answer: first_unique_answer"),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content='<code>\nfinal_answer("first_unique_answer")\n</code>',
            ),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content='<code>\nfinal_answer("second_unique_answer")\n</code>',
            ),
        ]
        with patch.object(agent.model, "generate", side_effect=messages) as generate:
            list(ui._stream_response("first_unique_task", []))
            self.assertIsNotNone(agent.memory.steps[1].error)
            list(ui._stream_response("second_unique_task", []))
        second_prompt = str(generate.call_args_list[-1].args[0])
        self.assertIn("second_unique_task", second_prompt)
        self.assertNotIn("first_unique_task", second_prompt)
        self.assertNotIn("first_unique_answer", second_prompt)
        self.assertTrue(all(getattr(step, "error", None) is None for step in agent.memory.steps))

    def test_agent_executes_tool_and_returns_observation(self):
        agent = create_agent("qwen2:7b", "http://localhost:11434")
        agent.logger.level = -1
        self.assertEqual(agent.model.model_id, "ollama_chat/qwen2:7b")
        self.assertIn("Never invent random results", agent.system_prompt)
        messages = [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Thought: Use the dice tool.\n<code>\nresult = roll_dice(min_rolls=2, max_rolls=10)\nprint(result)\n</code>",
            ),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Thought: Return the observed result.\n<code>\nfinal_answer(result)\n</code>",
            ),
        ]
        with patch.object(agent.model, "generate", side_effect=messages) as generate:
            with patch("app.random.randint", side_effect=[3, 2, 6, 1]):
                result = agent.run("Roll a random number of dice between 2 and 10 times.")
        self.assertEqual(result, {"count": 3, "rolls": [2, 6, 1], "total": 9})
        self.assertEqual(generate.call_count, 2)
        self.assertIn("'rolls': [2, 6, 1]", agent.memory.steps[1].observations)


if __name__ == "__main__":
    unittest.main()
