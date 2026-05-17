# Benchmark Prompt Examples

---

## Round 1 — Test 1a: Attribute Classification (5 prompts, English)

### Attribute 1: Epistemic Stance

```
You are a linguist specialising in colloquial Malay. Your task is to read the Malay sentence below and decide how certain the speaker sounds about the information they are conveying.
Referring to the following three labels and their definitions to make your decision:
Certain: The speaker treats the statement as already true or established. There is no hedging, no doubt, and no checking. The speaker is asserting the information with full confidence.
Uncertain: The speaker sounds unsure, is making a guess, is estimating, or is checking whether something is the case. Words like "agaknya" (I think/probably), "kot" (maybe), or question particles that probe for confirmation are typical signals.
Neutral/NA: The sentence does not carry any detectable certainty signal in either direction. This applies to neutral descriptions, commands, or sentences where certainty is simply not relevant.
Speaker: "{TEXT}"
Given the three labels "Certain, Uncertain, Neutral/NA", the most likely, single label of the speaker's utterance is:
```

---

### Attribute 2: Particle Position

```
You are a linguist specialising in colloquial Malay. Your task is to read the Malay sentence below and decide where the discourse particle appears in it.
Referring to the following three labels and their definitions to make your decision:
Front: The particle appears at the very start of the sentence, before any other content words.
Middle/End: The particle appears anywhere other than the front — mid-sentence, before the final word, or at the end.
N/A: No discourse particle is present in the sentence (e.g. the particle slot is shown as "[___]" or the sentence simply contains no particle).
Speaker: "{TEXT}"
Given the three labels "Front, Middle/End, N/A", the most likely, single label of the speaker's utterance is:
```

---

### Attribute 3: Listener Agreement

```
You are a linguist specialising in colloquial Malay. Your task is to read the Malay sentence below and decide how the speaker is orienting toward the listener in terms of shared knowledge or agreement.
Referring to the following three labels and their definitions to make your decision:
Assumed Agreement: The speaker treats the information as already shared or obvious to the listener. The sentence is presented as common ground — the underlying tone is "you already know this" or "of course this is true". No explicit confirmation is being requested.
Confirmation Seeking: The speaker is actively checking whether the listener agrees, knows, or can confirm the information. The sentence invites or requests the listener's validation before the speaker can proceed with confidence.
Neutral/Unclear: The sentence does not show any clear orientation toward listener agreement. This applies to plain statements, commands, or cases where the interpersonal stance toward agreement is ambiguous or absent.
Speaker: "{TEXT}"
Given the three labels "Assumed Agreement, Confirmation Seeking, Neutral/Unclear", the most likely, single label of the speaker's utterance is:
```

---

### Attribute 4: Emotion

```
You are a linguist specialising in colloquial Malay. Your task is to read the Malay sentence below and decide the emotional tone of the speaker.
Referring to the following three labels and their definitions to make your decision:
Positive: The speaker expresses happiness, excitement, enthusiasm, satisfaction, humour, affection, relief, or any other clearly positive feeling. This includes light-hearted teasing or playful sarcasm that is warm in tone.
Negative: The speaker expresses frustration, annoyance, disappointment, sadness, anger, bitterness, or any other clearly negative feeling. This includes hostile or bitter sarcasm.
Neutral/Unclear: The sentence carries no detectable emotional charge in either direction, or the emotion is genuinely ambiguous and cannot be reliably classified as positive or negative.
Speaker: "{TEXT}"
Given the three labels "Positive, Negative, Neutral/Unclear", the most likely, single label of the speaker's utterance is:
```

---

### Attribute 5: Question Type

```
You are a linguist specialising in colloquial Malay. Your task is to read the Malay sentence below and decide its primary sentence function.
Referring to the following three labels and their definitions to make your decision:
Declarative/Statement: The sentence makes an assertion or conveys information. It describes a situation, states a fact, or expresses a view. It is not structured as a question, even if it ends with a particle.
Rhetorical Interrogative: The sentence is phrased as a question but does not expect a genuine answer from the listener. It is used to make a point, express emotion, or emphasise something — the speaker already implies the answer through the question itself.
Yes/No Interrogative: The sentence is a genuine question that invites the listener to confirm or deny something. The speaker does not already know the answer and is seeking a real yes-or-no response.
Speaker: "{TEXT}"
Given the three labels "Declarative/Statement, Rhetorical Interrogative, Yes/No Interrogative", the most likely, single label of the speaker's utterance is:
```

---

## Round 1 — Test 2a: Attribute-Constrained Particle Generation (1 prompt)

**System message:**
```
You are a linguist specialising in colloquial Malay discourse particles. You must output exactly one word — either "ke" or "kan" or "neutral" — and nothing else.
```

**User message** (example with attributes: Epistemic Stance = Certain, Particle Position = Middle/End, Listener Agreement = Assumed Agreement, Emotion = Negative, Question Type = Rhetorical Interrogative):
```
You are given a Malay sentence in which one discourse particle has been replaced with [___].
Your task is to predict which particle, either "ke," "kan," or "neutral," belongs in the [___] slot, such that the discourse-context attributes for this sentence are The speaker treats the statement as already true or established — no hedging, no doubt, full confidence.
The particle appears anywhere other than the front — mid-sentence or at the end.
The speaker treats the information as shared or obvious — the underlying tone is 'you already know this'.
The speaker expresses frustration, annoyance, disappointment, sadness, anger, or bitterness.
Phrased as a question but does not expect a genuine answer; used to make a point or emphasise something.

Particle meanings:
  ke  : signals genuine uncertainty or invites the listener to confirm something the speaker is unsure about.
  kan : signals assumed shared knowledge and seeks light confirmation of something the speaker already believes ("right?").
  neutral : indicates no particle.

Speaker:
  "{TEXT}"



Using the sentence context and the attributes above, which single particle — "ke" or "kan" or "neutral" — best fills [___]?

Return exactly one word from this set and nothing else: ke, kan, neutral
```

---

## Round 1 — Test 2c: Unconstrained Baseline Particle Prediction (1 prompt)

```
You are a linguist specialising in colloquial Malay discourse particles. A discourse particle has been removed from the Malay sentence below and replaced with [___]. Your task is to predict which particle, either "ke" or "kan" or "neutral", belongs in that slot.

Particle meanings:
  ke  : signals genuine uncertainty or invites the listener to confirm something the speaker is unsure about.
  kan : signals assumed shared knowledge and seeks light confirmation of something the speaker already believes ("right?").
  neutral : indicates no particle.


Speaker: "{TEXT}"


Given the three candidate particles "kan" and "ke" and "neutral", the single most likely particle to fill [___] is:
```
