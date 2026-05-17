# Round 2 Benchmark Prompt Drafts

---

## Round 2 — Test 1b: Macro-Function Classification (Unassisted)

```
You are a linguist specialising in colloquial Malay discourse pragmatics. Your task is to read the Malay sentence below and identify the primary communicative role the utterance plays in interaction, beyond its literal propositional content.

Referring to the following seven labels and their definitions to make your decision:

Assumed-Agreement Rhetorical Stance: Speaker presents proposition as already obvious/shared knowledge; listener is expected to align rather than genuinely answer.

Neutral Declarative: Plain informational statements with minimal discourse pressure or stance marking.

Information-Seeking Verification: Genuine request for verification or clarification; speaker leaves room for disagreement.

Affective Confirmation-Seeking Question: Speaker seeks confirmation while simultaneously expressing affect (surprise, irritation, humour, excitement, disbelief, etc.)

Emphatic / Discourse-Marking: Particle functions less as a literal confirmation marker and more as a discourse-management or emphasis device.

Null Form Retaining Particle-Like Pragmatic Meaning: Pragmatic meaning associated with particles remains inferable even after overt particle removal.

Negative Rhetorical Challenge / Evaluation: Speaker uses rhetorical questioning to criticise, challenge, mock, or negatively evaluate a proposition rather than genuinely seek information.

Speaker: "{TEXT}"

Considering what the speaker is communicatively doing with this utterance, their stance, their orientation toward the listener, and the function the sentence serves in interaction, which of the seven labels best captures its primary discourse role?
The most likely, single label is:
```

---

## Round 2 — Test 1c: Macro-Function Classification (Attribute-Assisted)

**System message:**
```
You are a linguist specialising in colloquial Malay discourse pragmatics. You must output exactly one label from the provided list and nothing else.
```

**User message**:
```
You are a linguist specialising in colloquial Malay discourse pragmatics. Your task is to read the Malay sentence below and identify the primary communicative role the utterance plays in interaction, beyond its literal propositional content.

You are provided with the following human-annotated attribute labels for this sentence as additional context:
{attr_block}

Use these attributes to inform your decision, but base your final label on the overall communicative function of the utterance.

Referring to the following seven labels and their definitions to make your decision:

Assumed-Agreement Rhetorical Stance: Speaker presents proposition as already obvious/shared knowledge; listener is expected to align rather than genuinely answer.

Neutral Declarative: Plain informational statements with minimal discourse pressure or stance marking.

Information-Seeking Verification: Genuine request for verification or clarification; speaker leaves room for disagreement.


Affective Confirmation-Seeking Question: Speaker seeks confirmation while simultaneously expressing affect (surprise, irritation, humour, excitement, disbelief, etc.)

Emphatic / Discourse-Marking: Particle functions less as a literal confirmation marker and more as a discourse-management or emphasis device.

Null Form Retaining Particle-Like Pragmatic Meaning: Pragmatic meaning associated with particles remains inferable even after overt particle removal.

Speaker: "{TEXT}"

Considering what the speaker is communicatively doing with this utterance, their stance, their orientation toward the listener, and the function the sentence serves in interaction, which of the seven labels best captures its primary discourse role?
The most likely, single label is:
```

---

## Round 2 — Test 2b: Function-Constrained Particle Generation

**System message:**
```
You are a linguist specialising in colloquial Malay discourse particles. You must output exactly one word — either "ke" or "kan" or "neutral" — and nothing else.
```

**User message**:
```
You are given a Malay sentence in which one discourse particle has been replaced with [___].
Your task is to predict which particle — "ke", "kan", or "neutral" — belongs in the [___] slot, such that the sentence is consistent with the primary communicative role the utterance plays in interaction, beyond its literal propositional content:
{marco_function_attr_block}



Particle meanings:
  ke      : signals genuine uncertainty or invites the listener to confirm something the speaker is unsure about.
  kan     : signals assumed shared knowledge and seeks light confirmation of something the speaker already believes ("right?").
  neutral : indicates no particle.

Speaker:
  "{TEXT}"

Using the sentence context and the macro-function above, which single particle — "ke" or "kan" or "neutral" — best fills [___]?

Return exactly one word from this set and nothing else: ke, kan, neutral
```
