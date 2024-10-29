# ValiRel: Validation of Biomedical Relations using Lightweight LLMs

## Overview
ValiRel is a project that explores the effectiveness of lightweight large language models (LLMs), specifically Llama, for validating semantic biomedical relations between MeSH (Medical Subject Headings) keywords. Using this model, we assess and refine relations from a dataset generated by the [MeSH2Wikidata Project](https://figshare.com/articles/dataset/MeSH2Wikidata_A_set_of_tools_for_the_interaction_between_MeSH_keywords_OBO_Foundry_and_Wikidata_for_enriching_biomedical_knowledge/24438184), which includes semantic relations with different pointwise mutual information (PMI) scores.

### Repository Structure
The repository is organized into several directories:
- **`input`**: Contains the original dataset of semantic relations between MeSH keywords to be added to or removed from Wikidata, based on PMI values.
- **`calibration`**: Holds the calibration results, where verification repetitions are performed to determine the stable clustering point.
- **`add`**: Contains the refined list of new relations recommended for addition to Wikidata based on validation outcomes.
- **`verify`**: Contains the refined list of relations flagged for further verification or removal from Wikidata.

## Data Description
- **Input Data**: Includes two primary subsets:
  - **PMI < 2**: Wikidata relations between MeSH keywords with PMI scores below 2, indicating limited statistical association. This subset contains **35,719** relations.
  - **PMI >= 2 (Unlinked)**: MeSH keyword pairs with PMI scores of 2 or higher, lacking a corresponding relation in Wikidata. This subset contains **306,579** pairs.

## Experiments

### Calibration Experiment
In the initial experiment, we establish thresholds to allow the Llama model to reliably distinguish between true and false relations. A random sample of 500 biomedical relations with limited PMI and existing Wikidata relations were validated by querying the Llama model.

- **Number of Repetitions**: Each relation is queried 30 times to measure validation stability.
- **Result Metric (R)**: Calculated as \( R = \frac{T}{T+F} \), where \( T \) is the count of true classifications, and \( F \) is the count of false classifications.
- **Clustering Analysis**: We observed that after six repetitions, relations start to stabilize, achieving a final clustering based on Llama's assessments.

### Findings
- **True Relations**: Relations with an \( R(6) > 0.5 \) are considered true.
- **False Relations**: Relations with an \( R(6) < 0.866 \) are considered false.
- Full calibration results are documented in the `calibration` directory.

### Refinement of Relations
Based on calibration results, we refined our relation lists as follows:
1. **Relations to Add**: Relations with an \( R(6) > 0.5 \) are marked for potential addition to Wikidata. Results are stored in the `add` directory. Our experiment provided 209,827 refined pairs (68.4\%).
2. **Relations to Remove**: Relations with an \( R(6) < 0.866 \) are recommended for verification or potential removal from Wikidata. These are available in the `verify` directory. Our experiment provided (\%) refined relations requiring full attention.

## Repository Contents

- **`input/`**: Original dataset with semantic relations and PMI scores.
- **`calibration/`**: Calibration results showing Llama's validation outcome after six repetitions.
- **`add/`**: List of relations recommended for addition to Wikidata.
- **`verify/`**: List of relations flagged for verification or removal from Wikidata.

## Requirements

- **LLM Framework**: Ensure that the Llama model is set up and accessible.
- **Dependencies**: Required libraries for data handling and processing, including libraries for managing Llama queries.

## Usage

1. **Prepare the Dataset**: Ensure input data is available in the `input` directory.
2. **Run Calibration**: Execute the calibration experiment to observe clustering results after multiple verifications.
3. **Generate Output**: Based on calibration results, refine the relations and save them in the `add` and `verify` directories.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
