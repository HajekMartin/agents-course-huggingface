"""Verify tools, real vector-index plumbing, and CodeAct execution without services."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import ChatMessage, ChatResponse, MockLLM
from pydantic import PrivateAttr
from workflows.errors import WorkflowRuntimeError

from agent import conversation_messages, create_agent, guard_guest_answer, run_task
from executor import CodeExecutor
from retriever import GuestCatalog, GuestRetriever
from tools.guests import create_guest_tools
from tools.weather import get_weather
from tools.web import search_web
from diagnostics import request_id


class FixtureEmbedding(BaseEmbedding):
    """Deterministic vectors test the retrieval pipeline, not model quality."""
    model_name: str = "test-embedding"
    query_instruction: str = ""
    text_instruction: str = ""
    _text_calls: int = PrivateAttr(default=0)

    def _vector(self, text):
        if any(word in text.lower() for word in ("programming", "software", "computers")):
            return [1.0, 0.0, 0.0]
        if "radioactivity" in text.lower():
            return [0.0, 0.0, 1.0]
        return [0.0, 1.0, 0.0]

    def _get_text_embedding(self, text):
        self._text_calls += 1
        return self._vector(text)

    def _get_query_embedding(self, query):
        return self._vector(query)

    async def _aget_query_embedding(self, query):
        return self._vector(query)


class GuestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [
            {"name": "Ada Lovelace", "relation": "best friend", "description": "A mathematician who studied programming.", "email": "ada@example.com"},
            {"name": "Dr. Nikola Tesla", "relation": "old friend from university days", "description": "Researches wireless electricity.", "email": "tesla@example.com"},
            {"name": "Marie Curie", "relation": "no relation", "description": "Researches radioactivity.", "email": "curie@example.com"},
        ] + [
            {"name": f"Friend {i}", "relation": "friend", "description": "Enjoys biology.", "email": f"friend{i}@example.com"}
            for i in range(7)
        ]
        self.path = self.root / "guests.json"
        self.path.write_text(json.dumps({"guests": self.rows}))
        self.catalog = GuestCatalog(self.path)
        self.embed = FixtureEmbedding()
        self.retriever = GuestRetriever(self.catalog, self.embed, self.root / "index")
        self.functions = create_guest_tools(self.catalog, self.retriever)
        by_name = {function.__name__: function for function in self.functions}
        self.search = by_name["search_guests"]
        self.get_guest = by_name["get_guest"]
        self.filter_guests = by_name["filter_guests"]
        self.list_guests = by_name["list_guests"]
        self.find_name = by_name["find_guests_by_name"]

    def test_listing_returns_every_record_without_embeddings(self):
        result = self.list_guests()
        self.assertEqual(result["match_count"], len(self.rows))
        self.assertEqual(len(result["guests"]), len(self.rows))
        self.assertEqual(self.embed._text_calls, 0)

    def test_name_search_handles_fragments_typos_and_no_match(self):
        for name in ("Nicol", "Nik", "nikola", "Tesla"):
            with self.subTest(name=name):
                result = self.find_name(name)
                self.assertEqual([guest["name"] for guest in result["guests"]], ["Dr. Nikola Tesla"])
        self.assertEqual(self.find_name("Nicol")["match_type"], "fuzzy_spelling")
        self.assertEqual(self.find_name("Ada")["match_count"], 1)
        self.assertEqual(self.find_name("Zyzzyva")["guests"], [])
        self.assertEqual(self.find_name("Friend")["match_count"], 7)
        self.assertEqual(self.embed._text_calls, 0)

    def test_all_is_not_a_relation_category(self):
        with self.assertRaisesRegex(ValueError, "list_guests"):
            self.filter_guests("relation_category", "all")

    def test_semantic_ranking_returns_candidates_not_all_friends(self):
        result = self.search("software", top_k=1)
        self.assertFalse(result["exhaustive"])
        self.assertEqual(result["candidates"][0]["name"], "Ada Lovelace")
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["source"], f"{self.path.resolve()}#guest-0001")
        self.assertEqual(len(self.search("software", top_k=5)["candidates"]), 1)

    def test_filter_scans_entire_dataset_without_top_k_truncation(self):
        result = self.filter_guests("relation_category", " FRIEND ")
        self.assertTrue(result["exhaustive"])
        self.assertEqual(result["match_count"], 9)
        self.assertEqual(len(result["guests"]), 9)
        self.assertNotIn("Marie Curie", [guest["name"] for guest in result["guests"]])
        self.assertEqual(self.filter_guests("relation", "friend")["match_count"], 7)

    def test_lookup_normalizes_titles_but_does_not_guess_unknown_people(self):
        result = self.get_guest("  nikola   tesla ")
        self.assertTrue(result["found"])
        self.assertEqual(result["guest"]["email"], "tesla@example.com")
        self.assertFalse(self.get_guest("Unknown Person")["found"])
        self.assertFalse(self.get_guest("Ada")["found"])

    def test_invalid_search_and_filter_inputs(self):
        for query, top_k in (("", 1), ("science", 0), ("science", True), ("science", 21)):
            with self.subTest(query=query, top_k=top_k), self.assertRaises(ValueError):
                self.search(query, top_k)
        with self.assertRaises(ValueError):
            self.filter_guests("description", "biology")
        with self.assertRaises(ValueError):
            self.filter_guests("relation", "")
        for score in (True, -2, 2, float("nan")):
            with self.subTest(score=score), self.assertRaises(ValueError):
                self.search("software", min_score=score)
        self.assertEqual(self.filter_guests("relation_category", "family")["match_count"], 0)

    def test_persisted_index_reloads_without_reembedding_documents(self):
        self.search("software", 1)
        calls = self.embed._text_calls
        reloaded = GuestRetriever(self.catalog, self.embed, self.root / "index")
        self.assertEqual(reloaded.search("software", 1)["candidates"][0]["name"], "Ada Lovelace")
        self.assertEqual(self.embed._text_calls, calls)

    def test_index_key_changes_with_data_and_embedding_revision(self):
        original = self.retriever.index_dir
        revised = GuestRetriever(self.catalog, self.embed, self.root / "index", "different-model-digest")
        self.assertNotEqual(original, revised.index_dir)
        self.rows[0]["description"] = "Updated profile"
        self.path.write_text(json.dumps({"guests": self.rows}))
        changed = GuestRetriever(GuestCatalog(self.path), self.embed, self.root / "index")
        self.assertNotEqual(original, changed.index_dir)

    def test_duplicate_guest_names_are_rejected(self):
        self.rows.append({**self.rows[0], "name": "Lady Ada Lovelace"})
        self.path.write_text(json.dumps({"guests": self.rows}))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            GuestCatalog(self.path)

    def test_explicit_category_overrides_derived_category(self):
        self.rows[0]["relation_category"] = "family"
        self.path.write_text(json.dumps({"guests": self.rows}))
        self.assertEqual(GuestCatalog(self.path).guests[0]["relation_category"], "family")

    def test_executor_calls_tools_and_preserves_state_within_one_task(self):
        executor = CodeExecutor(self.functions)
        first = executor.execute('guest = get_guest(name="Ada Lovelace")\nprint(guest["guest"]["name"])')
        second = executor.execute('print(guest["guest"]["email"])')
        self.assertIn("Ada Lovelace", first["stdout"])
        self.assertIn("ada@example.com", second["stdout"])
        self.assertEqual(first["tool_calls"][0]["tool"], "get_guest")
        self.assertEqual(second["tool_calls"], [])
        other = CodeExecutor(self.functions)
        self.assertIn("error", other.execute('print(guest)'))

    def test_executor_keeps_request_id_in_worker_threads(self):
        from concurrent.futures import ThreadPoolExecutor

        def task_context() -> str:
            return request_id.get()

        token = request_id.set("test-request")
        try:
            executor = CodeExecutor(self.functions + [task_context])
        finally:
            request_id.reset(token)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.assertLogs("alfred", level="INFO"):
                result = pool.submit(executor.execute, 'print(task_context())').result()
                observed = pool.submit(request_id.get).result()
        self.assertNotIn("error", result)
        self.assertIn("test-request", result["stdout"])
        self.assertEqual(executor.request_id, "test-request")
        self.assertEqual(observed, "-")

    def test_executor_blocks_imports_private_access_and_tool_overwrites(self):
        for code in ("import os", "open('/tmp/agent-test-file', 'w')", "get_guest.__globals__", "get_guest = 3", "def get_guest():\n    pass"):
            with self.subTest(code=code):
                result = CodeExecutor(self.functions).execute(code)
                self.assertIn("error", result)
                self.assertEqual(result["tool_calls"], [])

    def test_tool_rules_and_examples_are_included_by_codeact(self):
        agent, _ = create_agent(MockLLM(), self.functions)
        descriptions = agent._get_tool_descriptions(agent.tools)
        self.assertIn("For all friends, use relation_category='friend'", descriptions)
        self.assertIn("Example task: Find guests interested", descriptions)
        self.assertNotIn("search_web", descriptions)


class ExternalToolTests(unittest.TestCase):
    def test_weather_is_explicitly_simulated(self):
        with patch("tools.weather.random.choice", return_value=("Rainy", 15)):
            output = get_weather("Turnov")
        self.assertTrue(output["simulated"])
        self.assertIn("not a live forecast", output["source"])

    def test_web_returns_source_urls_and_reports_network_errors(self):
        with patch("ddgs.DDGS") as client:
            client.return_value.text.return_value = [{"title": "Example", "href": "https://example.com/page", "body": "Snippet"}]
            result = search_web("public topic")
            self.assertEqual(result["results"][0]["url"], "https://example.com/page")
            client.return_value.text.return_value = [{}, {"href": "javascript:alert(1)"}]
            self.assertIn("no usable source links", search_web("public topic")["error"])
            client.return_value.text.side_effect = RuntimeError("offline")
            self.assertIn("offline", search_web("public topic")["error"])


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_console_logs_distinguish_disabled_web_and_model_steps(self):
        response = ChatResponse(message=ChatMessage(role="assistant", content="Enable web search first."))
        with self.assertLogs("alfred", level="INFO") as logs:
            with patch.object(MockLLM, "achat", new=AsyncMock(return_value=response)):
                async for _ in run_task(MockLLM(), [], "Search on the internet"):
                    pass
        output = "\n".join(logs.output)
        self.assertIn("web_enabled=False", output)
        self.assertIn("tools=[]", output)
        self.assertIn("WEB REQUESTED BUT DISABLED", output)
        self.assertIn("MODEL STEP 1", output)
        self.assertIn("FINAL ANSWER", output)
        self.assertIn("TASK END", output)

    async def test_explicit_web_request_gets_one_corrective_attempt(self):
        web = Mock(return_value={"results": [{"url": "https://example.com/ada", "snippet": "Ada was born in 1815."}]})
        def search_web(query: str, max_results: int = 5) -> dict:
            """Search public sources."""
            return web(query, max_results)
        responses = [
            ChatResponse(message=ChatMessage(role="assistant", content="Ada was born in 1815.")),
            ChatResponse(message=ChatMessage(role="assistant", content='<execute>\nprint(search_web(query="Ada Lovelace birth year"))\n</execute>')),
            ChatResponse(message=ChatMessage(role="assistant", content="Ada was born in 1815. Source: https://example.com/ada")),
        ]
        with patch.object(MockLLM, "achat", new=AsyncMock(side_effect=responses)) as chat:
            output = [item async for item in run_task(MockLLM(), [search_web], "Search her birth year on the internet")]
        web.assert_called_once()
        self.assertEqual(chat.await_count, 3)
        self.assertIn("corrective attempt", output[-1][1])
        self.assertIn("https://example.com/ada", output[-1][0])

    async def test_model_that_ignores_web_is_not_retried_forever(self):
        def search_web(query: str) -> dict:
            """Search public sources."""
            self.fail("The test model never actually calls this function")
        response = ChatResponse(message=ChatMessage(role="assistant", content="I cannot access the internet."))
        with patch.object(MockLLM, "achat", new=AsyncMock(return_value=response)) as chat:
            output = [item async for item in run_task(MockLLM(), [search_web], "Search online for Ada Lovelace")]
        self.assertEqual(chat.await_count, 2)
        self.assertIn("No web search was executed", output[-1][0])

    async def test_unsupported_birth_years_are_not_attributed_to_guest_records(self):
        calls = [{"tool": "list_guests", "result": {"guests": GuestCatalog().guests}}]
        for evidence in (calls, []):
            result = guard_guest_answer("Ada was born in 1815 according to data/guests.json.", evidence)
            self.assertIn("do not support", result)
            self.assertNotIn("1815", result)
            self.assertNotIn("Guest sources consulted", result)
        web = [{"tool": "search_web", "result": {"results": [{"url": "https://example.com/ada", "snippet": "Ada was born in 1815."}]}}]
        answer = "Ada was born in 1815. Source: https://example.com/ada"
        self.assertEqual(guard_guest_answer(answer, web), answer)

    async def test_follow_up_receives_history_without_previous_execution_state(self):
        history = [{"role": "user", "content": "Who are my guests?"},
                   {"role": "assistant", "content": "Ada Lovelace, Nikola Tesla and Marie Curie."}]
        response = ChatResponse(message=ChatMessage(role="assistant", content="Web search is disabled."))
        with patch.object(MockLLM, "achat", new=AsyncMock(return_value=response)) as chat:
            async for _ in run_task(MockLLM(), [], "Search their birth dates online", history=history):
                pass
        messages = str(chat.call_args.args[0])
        self.assertIn("Ada Lovelace", messages)
        self.assertIn("Search their birth dates online", messages)
        self.assertNotIn("Observation:", messages)

    async def test_history_is_bounded_and_supports_gradio_text_blocks(self):
        history = [{"role": "assistant", "content": "Working…"}] + [
            {"role": "user", "content": str(i) * 3000} for i in range(10)
        ]
        history.append({"role": "assistant", "content": [{"type": "text", "text": "Ada Lovelace"}]})
        messages = conversation_messages(history)
        self.assertEqual(len(messages), 8)
        self.assertTrue(all(len(message.content) <= 2000 for message in messages))
        self.assertEqual(messages[-1].content, "Ada Lovelace")

    async def test_empty_name_lookup_cannot_generate_placeholder_profiles(self):
        functions = create_guest_tools(GuestCatalog(), Mock())
        responses = [
            ChatResponse(message=ChatMessage(role="assistant", content='<execute>\nprint(find_guests_by_name(name="Zyzzyva"))\n</execute>')),
            ChatResponse(message=ChatMessage(role="assistant", content="1. [Candidate Profile 1] - [Score]")),
        ]
        with patch.object(MockLLM, "achat", new=AsyncMock(side_effect=responses)):
            output = [item async for item in run_task(MockLLM(), functions, "Find Zyzzyva")]
        self.assertNotIn("Candidate Profile", output[-1][0])
        self.assertIn("No matching guests", output[-1][0])

    async def test_placeholder_answer_uses_actual_returned_records(self):
        guest = GuestCatalog().guests[0]
        calls = [{"tool": "get_guest", "result": {"found": True, "guest": guest}}]
        result = guard_guest_answer("[Candidate Profile 1] - [Score]", calls)
        self.assertIn(guest["name"], result)
        self.assertIn(guest["source"], result)
        self.assertNotIn("[Score]", result)

    async def test_repeated_code_calls_stop_at_iteration_limit(self):
        response = ChatResponse(message=ChatMessage(role="assistant", content='<execute>\nprint("still working")\n</execute>'))
        with patch.object(MockLLM, "achat", new=AsyncMock(return_value=response)) as chat:
            with self.assertRaisesRegex(WorkflowRuntimeError, "Max iterations"):
                async for _ in run_task(MockLLM(), [], "Repeat forever", max_iterations=2):
                    pass
        self.assertLessEqual(chat.await_count, 2)

    async def test_real_codeact_loop_executes_python_then_returns_answer(self):
        catalog = GuestCatalog()
        retriever = Mock()
        functions = create_guest_tools(catalog, retriever)
        responses = [
            ChatResponse(message=ChatMessage(role="assistant", content='<execute>\nresult = filter_guests(field="relation_category", value="friend")\nprint(result)\n</execute>')),
            ChatResponse(message=ChatMessage(role="assistant", content="Ada and Tesla are friends. Sources: data/guests.json#guest-0001 and data/guests.json#guest-0002.")),
        ]
        with patch.object(MockLLM, "achat", new=AsyncMock(side_effect=responses)) as chat:
            output = [item async for item in run_task(MockLLM(), functions, "List all friends.")]
        self.assertEqual(chat.await_count, 2)
        answer, trace = output[-1]
        self.assertIn("Ada and Tesla", answer)
        self.assertIn("Step 2", trace)
        self.assertIn("filter_guests", trace)
        self.assertIn("match_count", trace)
        self.assertIn("Observation", trace)

    async def test_greeting_returns_without_executor(self):
        response = ChatResponse(message=ChatMessage(role="assistant", content="Hello!"))
        with patch.object(MockLLM, "achat", new=AsyncMock(return_value=response)) as chat:
            output = [item async for item in run_task(MockLLM(), [], "Hello")]
        self.assertEqual(chat.await_count, 1)
        self.assertEqual(output[-1][0], "Hello!")
        self.assertNotIn("Executing", output[-1][1])


if __name__ == "__main__":
    unittest.main()
