# F1 Duel — project guide

## The question, in one sentence
**Before a Grand Prix starts, can we predict who will finish ahead of whom better than the starting grid? And will the model be right on the real races still to come?**

This is not a presentation about F1: we ask one question, build a model, compare it with the simplest answer ("whoever starts ahead finishes ahead"), then with reality.

## The plan: why each part exists

| Part | Course method | Why we do it | What we find |
|---|---|---|---|
| **I. Data** | pandas | One verified source, and never use what happens during the race | 5,435 starts, 269 GPs, 2014–2026, checked against formula1.com |
| **II. Linear regression** | OLS, Ridge, Lasso, cross-validation | The simplest model: "what is a grid place worth?" | 1 grid place = 0.64 finishing places. But front places count more: pole wins 52% of races, P2 24%, P3 11%. So we add log(grid) |
| **III. PCA** | principal components | 7 team statistics that say almost the same thing. PCA summarises them | 2 axes are enough: **car performance** (76%) and **reliability** (13%) |
| **IV. k-means** | clustering, elbow, silhouette | The professor said "some teams simply have better cars". k-means groups them automatically | 3 families: top (84% top-10 rate), midfield (55%), back (20%) |
| **V. Logistic regression** | sigmoid, softmax, ROC/AUC | Our questions are probabilities. The same model answers three questions | Points: 7.87 of 10 found vs 7.71 for the grid. Duels: 77.4% vs 75.5% (significant). Winner: the favourite is the pole sitter in 86% of races, but the probabilities are reliable (when the model says 24%, it happens 24% of the time) |
| **VI. Verdict** | test on 2026, then live races | Compare with what really happened | The model beats the grid on duels in 2026 (81.2% vs 78.9%). Monaco: the grid wins there. The real test: Austin, Mexico, São Paulo, Las Vegas, Qatar, Abu Dhabi |

**Each part feeds the next:** linear regression shows the grid matters but is not linear (hence log(grid)); PCA creates the "car performance" axis; k-means creates the car families; logistic regression uses all of it.

## Parameters of the final model (know them for the oral)
8 features, all known on Saturday night:
1. `grid`: starting position
2. `log(grid)`: front-row advantage
3. `quali_gap`: gap to pole in qualifying (%)
4. `practice_gap`: gap to the best free-practice lap (%)
5. `car_perf`: PCA axis 1 (car performance)
6. `car_rel`: PCA axis 2 (reliability)
7. `tier_top`: the car is in the "top" family (k-means)
8. `tier_back`: the car is in the "back of the grid" family (k-means)

Duel: P(A ahead of B) = sigmoid(w · (x_A − x_B)). Weights w are in notebook 05; car performance and log(grid) matter most.

## What is honest (say it to the professor)
- The model is **never trained on the races it predicts**: each season is predicted from earlier seasons, each 2026 race from earlier races.
- **Before qualifying**, the model is worse than the grid: qualifying is the most important information.
- For **the winner**, the model's favourite is the pole sitter in 86% of races: we do not pick the winner better, but the chances are reliable (log-loss 1.37 vs 1.59 for the win rate of each grid slot).
- **At Monaco** the grid wins (9.2 vs 8.0 scorers); on the other grid-dependent tracks the model still wins.
- Top-10 probabilities are slightly too optimistic for outsiders (19% predicted vs 12% observed): a limitation to state.
- In 2026, with 17 races, the gain on points is not yet proven (the confidence interval contains 0). Live races will add data.

## The website (Technology with Impact, sessions 7–8)
- Flask = the model deployed as an API (`/api/duel`, `/api/race`), as in session 7.
- HTML/CSS/JS interface, as in session 8.
- "Ask the race engineer" = OpenAI API, as in the GenAI tutorial. It answers **only with the model's numbers** (it cannot invent statistics).
- Deployment: `app/DEPLOY.md`.

## Every race weekend
See `TODO.md` §3.

**Reminder:** the brief (§5.1) allows AI for code, requires **declaring all AI use** in every submission, and forbids AI-written report text. Write the report yourselves from this guide and the notebooks.
