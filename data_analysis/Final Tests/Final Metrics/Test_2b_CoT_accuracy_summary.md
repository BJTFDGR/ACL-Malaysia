# Test_2b_CoT — Overall Accuracy

| Model | Accuracy | Errors |
|---|---|---|
| claude-sonnet-4-6 | 0.7433 | 0 |
| claude-haiku-4-5 | 0.6791 | 0 |
| gpt-5 | 0.6791 | 0 |
| gpt-5.4-mini | 0.6791 | 0 |
| gemini-3.1-flash-lite | 0.6738 | 0 |
| gemini-3.1-pro-preview | 0.6738 | 0 |
| aisingapore/Gemma-SEA-LION-v4-27B-IT | 0.6578 | 0 |
| aisingapore/Llama-SEA-LION-v3.5-70B-R | 0.6578 | 0 |

# Test_2b_CoT — Accuracy by Macro_Function

| Model | Affective Confirmation-Seeking Question | Assumed-Agreement Rhetorical Stance | Emphatic / Discourse-Marking | Information-Seeking Verification | Negative Rhetorical Challenge / Evaluation | Neutral Declarative | Null Form Retaining Particle-Like Pragmatic Meaning |
|---|---|---|---|---|---|---|---|
| gpt-5 | 0.182 (n=11) | 0.727 (n=55) | 1.000 (n=15) | 0.667 (n=42) | 0.091 (n=11) | 0.891 (n=46) | 0.000 (n=7) |
| gpt-5.4-mini | 0.182 (n=11) | 0.727 (n=55) | 0.533 (n=15) | 0.690 (n=42) | 0.091 (n=11) | 0.978 (n=46) | 0.286 (n=7) |
| claude-sonnet-4-6 | 0.545 (n=11) | 0.727 (n=55) | 1.000 (n=15) | 0.690 (n=42) | 0.636 (n=11) | 0.913 (n=46) | 0.000 (n=7) |
| claude-haiku-4-5 | 0.182 (n=11) | 0.727 (n=55) | 0.467 (n=15) | 0.690 (n=42) | 0.091 (n=11) | 0.978 (n=46) | 0.429 (n=7) |
| gemini-3.1-pro-preview | 0.182 (n=11) | 0.745 (n=55) | 1.000 (n=15) | 0.595 (n=42) | 0.091 (n=11) | 0.891 (n=46) | 0.143 (n=7) |
| gemini-3.1-flash-lite | 0.182 (n=11) | 0.745 (n=55) | 1.000 (n=15) | 0.619 (n=42) | 0.091 (n=11) | 0.891 (n=46) | 0.000 (n=7) |
| aisingapore/Llama-SEA-LION-v3.5-70B-R | 0.182 (n=11) | 0.727 (n=55) | 0.600 (n=15) | 0.690 (n=42) | 0.091 (n=11) | 0.891 (n=46) | 0.143 (n=7) |
| aisingapore/Gemma-SEA-LION-v4-27B-IT | 0.182 (n=11) | 0.727 (n=55) | 0.800 (n=15) | 0.643 (n=42) | 0.273 (n=11) | 0.848 (n=46) | 0.000 (n=7) |
