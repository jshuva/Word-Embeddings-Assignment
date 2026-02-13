### Embedding Sensitivity Tests

To evaluate how sensitive our matchmaking pipeline is to the choice of the underlying embedding model, I replaced the default `all-MiniLM-L6-v2` model with `paraphrase-MiniLM-L6-v2`. I then calculated the cosine similarity between my profile and the rest of the class for both models, ranked the results, and compared them.

**Quantitative Impact:** The Spearman Rank Correlation between the two models' rankings is **0.8246**. This high positive correlation indicates that the overall, broad ordering of classmates—from most similar to least similar—remains largely consistent regardless of which model is used. Both models generally agree on the big picture.

**Qualitative Impact:** However, despite the strong overall correlation, the models are highly sensitive at the micro-level, significantly changing who my immediate "closest friends" are.

* Under the original model, my top three matches were: **Binziya Siddik, Mohammad Pakdoust, and Miguel Palafox**.
* Under the paraphrase model, my top three shifted to: **Mohammad Pakdoust, Mohamed Drira, and Md Musfiqur Rahman**.

While Mohammad Pakdoust remained a top match across both models, the rest of my top matches changed completely. This reveals that while different embedding models agree on general similarities, the fine-grained rankings at the very top are highly sensitive to the specific model's architecture and training data. The choice of model undeniably alters the final matchmaking results.