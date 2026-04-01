# Scaling Evaluation to 100k+ Jobs with Synthetic Ground Truth

The core challenge in evaluating a content-based recommendation engine like yours is the lack of "user interaction" data (e.g., we don't know which users clicked on which jobs). Because you only have a `job_descriptions.csv`, we must construct a **Synthetic Ground Truth** to calculate objective metrics like Precision@10, Recall@10, and NDCG@10 across your massive 100k+ dataset.

## Proposed Strategy: "Leave-One-Out" Simulated Candidates

Instead of manually handwriting 10 test cases, we will automatically generate thousands of test cases directly from the data itself.

1. **Simulate Resumes:** We randomly sample $N$ (e.g., 500) jobs from the 100k dataset. We treat the `Job Description` and `skills` of these sampled jobs as "Simulated Resumes".
2. **Define Relevance (Ground Truth):** For a simulated resume derived from a "Data Scientist" job, we define a recommended job as **Relevant (1)** if it shares the exact same `Job Title` as the source job. It is **Irrelevant (0)** otherwise.
3. **Execute:** The recommendation engine runs these 500 "resumes" against the entire 100k+ pre-computed dataset (`jobs.pkl`, `bert_job_vectors.pkl`, etc.).
4. **Calculate Metrics:**
   - **Precision@10:** What percentage of the top 10 recommended jobs have the target Job Title?
   - **Recall@10:** Out of *all* jobs in the 100k dataset with that exact Job Title, what percentage did we catch in the top 10?
   - **NDCG@10:** A ranking metric. If the relevant matches appear at rank #1 and #2, the score is higher than if they appear at rank #9 and #10.

## User Review Required

> [!IMPORTANT]
> Because you trained your model on a **Cloud GPU** using Colab, the 100k+ models (`bert_job_vectors.pkl`, `job_vectors.pkl`, `jobs.pkl`) are massive.
> 
> **Are these artifacts currently downloaded and resting in your local `"artifacts/"` folder?** 
> * The evaluation script will fail if it cannot load the 500,000+ vector rows you just trained.
> * If they are present, we can proceed to evaluate the 100k model. Otherwise, you must download them from the Colab VS Code explorer first.

## Proposed Changes

We will restructure `evaluate.py` to support two modes: the manual sanity checks, and the large-scale synthetic automated test.

### 1. Re-writing `evaluate.py`

#### [MODIFY] `evaluate.py`
- Formulate the synthetic candidate generator (`generate_synthetic_test_cases(jobs_df, sample_size=500)`).
- Implement mathematical definitions for `precision_at_k`, `recall_at_k`, and `ndcg_at_k`.
- Update the artifact loader to load the massive unified `bert_job_vectors.pkl` trained from your Colab session.
- Add an execution flag to run the `Automated Synthetic Evaluation` which will output a detailed markdown/console report of the engine's real-world accuracy on the 100k dataset.

## Open Questions

1. **How many test cases would you like to use?** A sample of 500 jobs across 100k data is statistically significant, but can take ~1-2 minutes to calculate locally. 
2. **Do you agree with using "Exact Job Title" as the metric for relevance?** (e.g., If the query is derived from a "Python Developer" job, only returning "Python Developer" counts as a hit. Returning "Software Engineer" would count as a miss. This is strict, but highly objective).
