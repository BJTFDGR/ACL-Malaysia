# Test_2d — Overall Accuracy

| Model | Accuracy | Errors |
|---|---|---|
| claude-haiku-4-5 | 0.7112 | 0 |
| claude-sonnet-4-6 | 0.7059 | 0 |
| gemini-3.1-pro-preview | 0.6898 | 0 |
| gpt-5 | 0.6898 | 0 |
| gpt-5.4-mini | 0.6898 | 0 |
| gemini-3.1-flash-lite | 0.6791 | 0 |
| aisingapore/Gemma-SEA-LION-v4-27B-IT | 0.5615 | 0 |
| aisingapore/Llama-SEA-LION-v3.5-70B-R | 0.3369 | 0 |

# Test_2d — Accuracy by Macro_Function

| Model | Affective Confirmation-Seeking Question | Assumed-Agreement Rhetorical Stance | Emphatic / Discourse-Marking | Information-Seeking Verification | Negative Rhetorical Challenge / Evaluation | Neutral Declarative | Null Form Retaining Particle-Like Pragmatic Meaning |
|---|---|---|---|---|---|---|---|
| gpt-5 | 0.636 (n=11) | 0.745 (n=55) | 0.600 (n=15) | 0.690 (n=42) | 0.364 (n=11) | 0.848 (n=46) | 0.000 (n=7) |
| gpt-5.4-mini | 0.545 (n=11) | 0.727 (n=55) | 0.667 (n=15) | 0.690 (n=42) | 0.273 (n=11) | 0.870 (n=46) | 0.143 (n=7) |
| claude-sonnet-4-6 | 0.636 (n=11) | 0.818 (n=55) | 0.933 (n=15) | 0.667 (n=42) | 0.818 (n=11) | 0.630 (n=46) | 0.000 (n=7) |
| claude-haiku-4-5 | 0.636 (n=11) | 0.745 (n=55) | 1.000 (n=15) | 0.690 (n=42) | 0.273 (n=11) | 0.826 (n=46) | 0.000 (n=7) |
| gemini-3.1-pro-preview | 0.636 (n=11) | 0.745 (n=55) | 1.000 (n=15) | 0.667 (n=42) | 0.636 (n=11) | 0.652 (n=46) | 0.143 (n=7) |
| gemini-3.1-flash-lite | 0.636 (n=11) | 0.745 (n=55) | 1.000 (n=15) | 0.667 (n=42) | 0.636 (n=11) | 0.630 (n=46) | 0.000 (n=7) |
| aisingapore/Llama-SEA-LION-v3.5-70B-R | 0.273 (n=11) | 0.000 (n=55) | 0.000 (n=15) | 0.167 (n=42) | 0.091 (n=11) | 1.000 (n=46) | 0.857 (n=7) |
| aisingapore/Gemma-SEA-LION-v4-27B-IT | 0.636 (n=11) | 0.709 (n=55) | 1.000 (n=15) | 0.690 (n=42) | 0.545 (n=11) | 0.196 (n=46) | 0.000 (n=7) |
