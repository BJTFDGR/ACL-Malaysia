# Round 3 Benchmark Prompt Drafts

Three new tests:
1. **Test 2d** — Particle generation given **both** attribute breakdown **and** macro-function.
2. **Test 1b-CoT** — Macro-function classification with chain-of-thought, **no attributes**.
3. **Test 2b-CoT** — Particle generation with chain-of-thought, given macro-function only.

For 1b-CoT and 2b-CoT the model is asked to emit a reasoning trace followed by a final answer line. The full response (reasoning + answer) is saved to the IO logs and to a `{model}_reasoning` column in the predictions CSV. The label is parsed from the `Final answer:` line.

---

## Test 2d — Attribute + Function-Constrained Particle Generation

**System message:**
```
You are a linguist specialising in colloquial Malay discourse particles. You must output exactly one word — either "ke", "kan", or "neutral" — and nothing else.
```

**User prompt template:**
```
You are given a Malay sentence in which one discourse particle has been replaced with [___].

Discourse-context attributes for this sentence:
- Epistemic Stance: {ES_VAL} — {ES_DESC}
- Particle Position: {PP_VAL} — {PP_DESC}
- Listener Agreement: {LA_VAL} — {LA_DESC}
- Emotion: {EM_VAL} — {EM_DESC}
- Question Type: {QT_VAL} — {QT_DESC}

Macro-pragmatic function of this utterance:
{MACRO_FUNCTION}: {MACRO_FUNCTION_DEFINITION}

Particle meanings:
  ke      : signals genuine uncertainty or invites the listener to confirm something the speaker is unsure about.
  kan     : signals assumed shared knowledge and seeks light confirmation of something the speaker already believes ("right?").
  neutral : indicates no particle.

Speaker:
  "{TEXT_MASKED}"

Using both the attribute breakdown and the macro-function above, which single particle — "ke", "kan", or "neutral" — best fills [___]?

Return exactly one word from this set and nothing else: ke, kan, neutral
```

---

## Test 1b-CoT — CoT Macro-Function Classification (No Attributes)

**System message:**
```
You are a linguist specialising in colloquial Malay discourse pragmatics. Reason step-by-step about the speaker's communicative intent, then output exactly one macro-function label from the provided list.
```

**User prompt template:**
```
You are given a Malay sentence. Your task is to classify its primary macro-pragmatic function — the communicative role the utterance plays in interaction, beyond its literal propositional content.

Referring to the following seven labels and their definitions:

Assumed-Agreement Rhetorical Stance: Speaker presents proposition as already obvious/shared knowledge; listener is expected to align rather than genuinely answer.

Neutral Declarative: Plain informational statements with minimal discourse pressure or stance marking.

Information-Seeking Verification: Genuine request for verification or clarification; speaker leaves room for disagreement.

Affective Confirmation-Seeking Question: Speaker seeks confirmation while simultaneously expressing affect (surprise, irritation, humour, excitement, disbelief, etc.).

Emphatic / Discourse-Marking: Particle functions less as a literal confirmation marker and more as a discourse-management or emphasis device.

Null Form Retaining Particle-Like Pragmatic Meaning: Pragmatic meaning associated with particles remains inferable even after overt particle removal.

Negative Rhetorical Challenge / Evaluation: Speaker uses rhetorical questioning to criticise, challenge, mock, or negatively evaluate a proposition rather than genuinely seek information.

Speaker: "{TEXT}"

Think step-by-step:
  1. What is the speaker doing in this utterance (asserting, asking, hedging, emphasising, challenging)?
  2. What is the speaker's stance toward the listener (shared knowledge, seeking confirmation, neutral information, etc.)?
  3. Which of the seven labels best captures this combination?

Format your response EXACTLY as follows, with no extra text before or after:

Reasoning: <2–4 sentences of step-by-step analysis>
Final answer: <one label from the list above>
```

---

## Test 2b-CoT — CoT + Function-Constrained Particle Generation

**System message:**
```
You are a linguist specialising in colloquial Malay discourse particles. Reason step-by-step, then output exactly one particle — "ke", "kan", or "neutral".
```

**User prompt template:**
```
You are given a Malay sentence in which one discourse particle has been replaced with [___].

The macro-pragmatic function of this utterance is:
  {MACRO_FUNCTION}: {MACRO_FUNCTION_DEFINITION}

Particle meanings:
  ke      : signals genuine uncertainty or invites the listener to confirm something the speaker is unsure about.
  kan     : signals assumed shared knowledge and seeks light confirmation of something the speaker already believes ("right?").
  neutral : indicates no particle.

Speaker:
  "{TEXT_MASKED}"

Think step-by-step:
  1. What is the speaker communicatively doing, given the macro-function above?
  2. Does the masked slot call for genuine uncertainty (ke), assumed shared knowledge (kan), or no particle (neutral)?
  3. Which choice best fits both the sentence flow and the macro-function?

Format your response EXACTLY as follows, with no extra text before or after:

Reasoning: <2–4 sentences of step-by-step analysis>
Final answer: <ke | kan | neutral>
```

---

## Parsing notes

- **Particle extraction** (Tests 2d, 2b-CoT): regex `(?i)final\s*answer:\s*(\S+)` → lowercase → match against `{ke, kan, neutral}`. Fallback to existing `_extract_particle` if the `Final answer:` line is missing.
- **Macro-function extraction** (Test 1b-CoT): regex `(?i)final\s*answer:\s*(.+?)(?:\n|$)` → strip → match against the 7 canonical labels (case-insensitive exact / substring / closest match).
- **Reasoning storage**: the full raw response (reasoning trace + final-answer line) is written to a new column `{model_name}_reasoning` in the predictions CSV, and also embedded in the IO log `raw_output` field.

---

## Reporting

For each test we produce:
1. Overall accuracy per model (markdown + CSV).
2. Per-Macro-Function breakdown (accuracy stratified by ground-truth macro-function label).
3. A combined ablation table comparing:
   - **For particle generation:** 2a (attributes) vs 2b (function) vs 2b-CoT vs 2d (both).
   - **For function prediction:** 1b (unassisted) vs 1b-CoT vs 1c (attribute-assisted).
