#!/usr/bin/python3
"""Scanner checks for the Grok usage collector."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

PLUGIN = Path(__file__).resolve().parents[1]
loader = SourceFileLoader("grok_collector", str(PLUGIN / "omarchy-agent-usage-grok"))
spec = importlib.util.spec_from_loader("grok_collector", loader)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["grok_collector"] = mod
spec.loader.exec_module(mod)


class TurnUsageTests(unittest.TestCase):
  def test_splits_cache_out_of_input(self) -> None:
    input_tokens, output_tokens, cache_read, cache_write = mod.turn_usage({
      "inputTokens": 100,
      "outputTokens": 20,
      "cachedReadTokens": 30,
      "cacheCreationTokens": 10,
      "reasoningTokens": 5,
    })
    self.assertEqual((input_tokens, output_tokens, cache_read, cache_write), (60, 20, 30, 10))

  def test_reasoning_lifts_output(self) -> None:
    _, output_tokens, _, _ = mod.turn_usage({
      "inputTokens": 10,
      "outputTokens": 2,
      "reasoningTokens": 8,
    })
    self.assertEqual(output_tokens, 8)


class ScanUpdatesTests(unittest.TestCase):
  def test_counts_turn_completed_and_skips_subagent(self) -> None:
    with tempfile.TemporaryDirectory() as raw:
      root = Path(raw)
      parent = root / "parent"
      child = root / "child"
      parent.mkdir()
      child.mkdir()
      (parent / "summary.json").write_text(json.dumps({"session_kind": "local"}), encoding="utf-8")
      (child / "summary.json").write_text(json.dumps({"session_kind": "subagent"}), encoding="utf-8")
      turn = {
        "timestamp": "2026-08-28T12:00:00Z",
        "params": {
          "update": {
            "sessionUpdate": "turn_completed",
            "prompt_id": "p1",
            "model": "grok-4.6-build",
            "usage": {
              "inputTokens": 50,
              "outputTokens": 10,
              "cachedReadTokens": 20,
            },
          }
        },
      }
      (parent / "updates.jsonl").write_text(json.dumps(turn) + "\n", encoding="utf-8")
      (child / "updates.jsonl").write_text(json.dumps(turn) + "\n", encoding="utf-8")

      stats = mod.scan_sessions(root)
      self.assertEqual(stats["totalPrompts"], 1)
      self.assertEqual(stats["totalSessions"], 1)
      self.assertEqual(stats["modelUsage"]["grok-4.6-build"]["inputTokens"], 30)
      self.assertEqual(stats["modelUsage"]["grok-4.6-build"]["cacheReadInputTokens"], 20)
      self.assertEqual(stats["modelUsage"]["grok-4.6-build"]["outputTokens"], 10)


class BillingParseTests(unittest.TestCase):
  def test_weekly_and_build_skip_voice(self) -> None:
    parsed = mod.parse_billing({
      "config": {
        "creditUsagePercent": 7.0,
        "currentPeriod": {
          "type": "USAGE_PERIOD_TYPE_WEEKLY",
          "end": "2026-08-31T22:15:50+00:00",
        },
        "productUsage": [
          {"product": "GrokVoice", "usagePercent": 6.0},
          {"product": "GrokBuild", "usagePercent": 1.0},
        ],
        "prepaidBalance": {"val": 0},
        "onDemandCap": {"val": 0},
        "onDemandUsed": {"val": 0},
      }
    }, "SuperGrok")
    labels = [entry["title"] for entry in parsed["limits"]]
    self.assertEqual(labels, ["Weekly", "Grok Build Weekly"])
    self.assertAlmostEqual(parsed["limits"][0]["percent"], 0.07)
    self.assertAlmostEqual(parsed["limits"][1]["percent"], 0.01)
    self.assertIsNone(parsed["balance"])
    self.assertEqual(parsed["tierLabel"], "SuperGrok")

  def test_prepaid_balance(self) -> None:
    parsed = mod.parse_billing({
      "config": {
        "creditUsagePercent": 0,
        "prepaidBalance": {"val": 12.5},
        "onDemandCap": {"val": 0},
        "onDemandUsed": {"val": 0},
      }
    }, "SuperGrok")
    self.assertEqual(parsed["balance"]["remaining"], 12.5)
    self.assertEqual(parsed["balance"]["funded"], 12.5)


if __name__ == "__main__":
  unittest.main()
