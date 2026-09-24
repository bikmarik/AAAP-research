# Autoencoder Asset Pricing Models — Presentation Plan

## Slide 1 — Research Question

**Can nonlinear factor exposures improve asset return prediction?**

Show the contrast immediately:

\[
\text{IPCA: } \beta(z) = z^\top \Gamma
\]

vs.

\[
\text{Paper: } \beta(z) = NN(z)
\]

### Goal
Make the audience understand the paper's motivation in the first minute.

---

## Slide 2 — Dataset Overview

Show one simple timeline:

```text
1957────────1974 | 1975────1986 | 1987────────────────2016
     TRAIN              VALIDATION             TEST
```

### Key numbers

- Monthly US stock returns
- NYSE, AMEX, NASDAQ
- ~30,000 stocks
- 94 stock characteristics
- 1957–2016
- Temporal train / validation / test split
- Models refitted annually

Do not list all 94 characteristics.

---

## Slide 3 — Solution Proposed by the Paper

Use one equation:

\[
r_{i,t}
=
\beta(z_{i,t-1})^\top f_t
+
u_{i,t}
\]

Then highlight the actual contribution:

```text
Old:

characteristics
      ↓
linear mapping
      ↓
factor exposures β


Paper:

characteristics
      ↓
neural network
      ↓
factor exposures β
```

### Main message

The paper keeps the conditional factor model but replaces the linear characteristics-to-beta relationship with a nonlinear neural network.

---

## Slide 4 — Architecture

Use a simplified architecture diagram:

```text
Characteristics z(t-1)
        │
        ▼
   Neural Network
        │
        ▼
 Conditional β(t-1)
        │
        │
        ▼
      βᵀf
        │
        ▼
   Asset Returns
        ▲
        │
 Latent Factors f(t)
        ▲
        │
Characteristic-Managed
     Portfolios
```

### Explain verbally

There are basically two sides:

1. **Beta network**
   - stock characteristics
   - neural network
   - conditional factor exposures

2. **Factor network**
   - returns / managed portfolios
   - compressed into latent factors

Then:

\[
\hat r = \beta^\top f
\]

---

## Slide 5 — Does It Work?

Use the paper's **predictive \(R^2\)** results.

| Model | Best Predictive \(R^2\) |
|---|---:|
| PCA | < 0 |
| IPCA | ~0.31% |
| CA0 | ~0.27% |
| CA1 | ~0.56% |
| CA2 | ~0.58% |
| CA3 | ~0.57% |

### Main message

Nonlinear conditional autoencoders roughly double the predictive \(R^2\) of IPCA.

Important: the absolute \(R^2\) is small because individual monthly stock returns are extremely noisy. The comparison between models is what matters.

---

## Slide 6 — Does It Make Money?

Explain the portfolio experiment:

```text
Predicted returns
       ↓
Rank stocks
       ↓
10 deciles
       ↓

LONG highest predicted returns
SHORT lowest predicted returns
```

Or:

\[
\text{Portfolio}
=
\text{Decile 10}
-
\text{Decile 1}
\]

Then show the Sharpe-ratio result from the paper.

### Main message

Better statistical predictions also translate into stronger long-short portfolio performance.

Mention that a real implementation would still need to account for:

- transaction costs
- turnover
- spreads
- liquidity
- market impact

---

## Slide 7 — Critique

Use exactly **three critiques**.

### 1. Missing Generic Nonlinear Baseline

The paper compares against factor models, but an obvious question is:

**Does the factor structure actually help?**

Compare:

\[
z_{i,t-1}
\rightarrow
MLP
\rightarrow
\hat r_{i,t}
\]

against:

\[
z_{i,t-1}
\rightarrow
NN
\rightarrow
\beta
\rightarrow
\beta^\top f
\]

Maybe a generic nonlinear predictor performs similarly.

### 2. Nonlinearity Can Overfit

Their own simulations show that when the true relationship is linear, IPCA can outperform the more flexible nonlinear models.

So:

\[
\text{more flexibility}
\neq
\text{automatically better}
\]

The usefulness of the neural network depends on whether the underlying relationship is actually nonlinear.

### 3. Portfolio Performance Is Not the Same as an Implementable Strategy

The paper reports strong portfolio results, but a real strategy must deal with:

- transaction costs
- spreads
- turnover
- liquidity
- market impact

A beautiful Sharpe ratio can get absolutely mugged by execution costs.

---

## Slide 8 — Our Experiment

This should directly answer Critique #1.

### Research Question

**Does the conditional factor structure add predictive value beyond generic nonlinear prediction?**

### Models

```text
Same dataset
    │
    ├── CA0
    │   Linear conditional factor model
    │
    ├── CA1
    │   Nonlinear conditional factor model
    │
    └── Direct MLP
        Characteristics → future return
```

### Evaluation

Use the same:

- train period
- validation period
- test period
- stock universe
- features
- evaluation metrics

Compare:

\[
R^2_{\text{pred}}
\]

and optionally:

\[
\text{Sharpe Ratio}
\]

### Main experiment

\[
\boxed{
CA0
\quad vs \quad
CA1
\quad vs \quad
Direct\ MLP
}
\]

If CA1 beats MLP:

> The economic factor structure appears useful.

If MLP performs similarly or better:

> The improvement may primarily come from nonlinear prediction rather than the factor-model structure.

Both outcomes are interesting.

---

## Slide 9 — Conclusion

Keep this brutally short.

1. Conditional autoencoders replace the linear characteristics-to-beta relationship with a nonlinear neural network.
2. The paper reports improved out-of-sample prediction and portfolio performance relative to traditional factor models.
3. The remaining question is whether the **factor structure itself** adds value beyond generic nonlinear machine learning.

---

## Presentation Flow

```text
QUESTION
   ↓
DATA
   ↓
SOLUTION
   ↓
ARCHITECTURE
   ↓
STATISTICAL RESULT
   ↓
ECONOMIC RESULT
   ↓
CRITIQUE
   ↓
OUR EXPERIMENT
   ↓
CONCLUSION
```

## Timing

| Part | Time |
|---|---:|
| Slides 1–2 | ~2 min |
| Slides 3–4 | ~3 min |
| Slides 5–6 | ~5 min |
| Slide 7 | ~3 min |
| Slides 8–9 | ~2 min |

**9 slides / 15 minutes is basically perfect.**

Keep Slide 9 on screen during questions instead of wasting a slide on “Thank you”.
