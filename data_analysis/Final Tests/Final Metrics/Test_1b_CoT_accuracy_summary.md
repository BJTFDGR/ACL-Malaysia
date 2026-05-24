# Test_1b_CoT — Overall Accuracy

| Model | Accuracy | Errors |
|---|---|---|
| gpt-5 | 0.3957 | 0 |
| deepseek-v4-flash | 0.3636 | 0 |
| claude-haiku-4-5 | 0.3476 | 0 |
| gemini-3.1-pro-preview | 0.3476 | 0 |
| gemini-3.1-flash-lite | 0.3422 | 0 |
| deepseek-v4-pro | 0.3262 | 0 |
| claude-sonnet-4-6 | 0.3209 | 0 |
| gpt-5.4-mini | 0.3155 | 0 |
| aisingapore/Gemma-SEA-LION-v4-27B-IT | 0.2620 | 0 |
| aisingapore/Llama-SEA-LION-v3.5-70B-R | 0.2193 | 0 |

# Test_1b_CoT — Accuracy by Macro_Function

| Model | Affective Confirmation-Seeking Question | Assumed-Agreement Rhetorical Stance | Emphatic / Discourse-Marking | Information-Seeking Verification | Negative Rhetorical Challenge / Evaluation | Neutral Declarative | Null Form Retaining Particle-Like Pragmatic Meaning |
|---|---|---|---|---|---|---|---|
| gpt-5 | 0.364 (n=11) | 0.418 (n=55) | 0.200 (n=15) | 0.643 (n=42) | 0.909 (n=11) | 0.152 (n=46) | 0.000 (n=7) |
| gpt-5.4-mini | 0.091 (n=11) | 0.364 (n=55) | 0.133 (n=15) | 0.452 (n=42) | 0.818 (n=11) | 0.174 (n=46) | 0.000 (n=7) |
| claude-sonnet-4-6 | 0.273 (n=11) | 0.400 (n=55) | 0.067 (n=15) | 0.476 (n=42) | 0.636 (n=11) | 0.087 (n=46) | 0.429 (n=7) |
| claude-haiku-4-5 | 0.182 (n=11) | 0.491 (n=55) | 0.133 (n=15) | 0.548 (n=42) | 0.636 (n=11) | 0.087 (n=46) | 0.000 (n=7) |
| gemini-3.1-pro-preview | 0.091 (n=11) | 0.527 (n=55) | 0.000 (n=15) | 0.500 (n=42) | 0.727 (n=11) | 0.130 (n=46) | 0.000 (n=7) |
| gemini-3.1-flash-lite | 0.091 (n=11) | 0.527 (n=55) | 0.000 (n=15) | 0.476 (n=42) | 0.727 (n=11) | 0.130 (n=46) | 0.000 (n=7) |
| deepseek-v4-pro | 0.091 (n=11) | 0.382 (n=55) | 0.000 (n=15) | 0.548 (n=42) | 0.818 (n=11) | 0.109 (n=46) | 0.286 (n=7) |
| deepseek-v4-flash | 0.091 (n=11) | 0.545 (n=55) | 0.000 (n=15) | 0.452 (n=42) | 0.818 (n=11) | 0.196 (n=46) | 0.000 (n=7) |
| aisingapore/Llama-SEA-LION-v3.5-70B-R | 0.091 (n=11) | 0.309 (n=55) | 0.000 (n=15) | 0.190 (n=42) | 0.909 (n=11) | 0.109 (n=46) | 0.000 (n=7) |
| aisingapore/Gemma-SEA-LION-v4-27B-IT | 0.273 (n=11) | 0.436 (n=55) | 0.133 (n=15) | 0.238 (n=42) | 0.727 (n=11) | 0.043 (n=46) | 0.000 (n=7) |
