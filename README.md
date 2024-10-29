# ValiRel: Validation of Biomedical Relations using Lightweight LLMs

## Overview
ValiRel evaluates the accuracy and utility of lightweight large language models (LLMs), specifically Llama, for validating semantic biomedical relations between Medical Subject Headings (MeSH) keywords. Leveraging this model, we refine relations within a dataset generated by the [MeSH2Wikidata Project](https://figshare.com/articles/dataset/MeSH2Wikidata_A_set_of_tools_for_the_interaction_between_MeSH_keywords_OBO_Foundry_and_Wikidata_for_enriching_biomedical_knowledge/24438184), which includes various semantic relations and pointwise mutual information (PMI) scores.

### Repository Structure
The repository is organized as follows:
- **`input`**: Original dataset of MeSH semantic relations, to be validated and potentially added to or removed from Wikidata, based on PMI scores.
- **`calibration`**: Results of the calibration experiments, which establish model stability for relation verification.
- **`add`**: Contains refined lists of new relations recommended for addition to Wikidata.
- **`verify`**: Lists of relations recommended for further verification or removal from Wikidata.

## Data Description
The input data includes two primary subsets:
  - **PMI < 2**: Wikidata relations between MeSH keywords with PMI scores below 2, indicating limited statistical association. This subset contains **35,719** relations.
  - **PMI ≥ 2 (Unlinked)**: MeSH keyword pairs with PMI scores of 2 or higher but lacking a relation in Wikidata. This subset includes **306,579** pairs.

## Experiments

### Calibration Experiment
The initial experiment determines thresholds that allow Llama to differentiate true relations from false ones. A random sample of 500 biomedical relations, with low PMI scores and existing Wikidata relations, was used for validation by querying the Llama model.

- **Repetition Count**: Each relation undergoes 30 validation repetitions to gauge stability.
- **Result Metric (R)**: Defined as $R = \frac{T}{T+F}$, where $T$ is the number of true classifications, and $F$ is the number of false classifications.
- **Clustering Analysis**: Stability in relation classification was achieved after six repetitions, allowing final clustering based on Llama's assessments.

### Findings
- **True Relations**: Relations with $R(6) > 0.5$ are designated as true.
- **False Relations**: Relations with $R(6) < 0.866$ are designated as false.
- Detailed calibration results are located in the `calibration` directory.

### Relation Refinement
Based on the calibration, relations were refined into the following lists:
1. **Additions**: Relations with $R(6) > 0.5$ are proposed for Wikidata addition. This subset contains **209,827** pairs (68.4% of tested relations).
2. **Verifications**: Relations with $R(6) < 0.866$ are marked for further verification or potential removal from Wikidata. This subset contains **12,183** pairs (34.1% of tested relations).

## Repository Contents

- **`input/`**: Contains the original MeSH relations dataset with PMI scores.
- **`calibration/`**: Documents Llama's validation stability results after six repetitions.
- **`add/`**: Stores relations suggested for addition to Wikidata.
- **`verify/`**: Contains relations flagged for further verification or removal.

## Requirements

- **LLM Framework**: Ensure that the Llama model is installed and accessible. We used https://huggingface.co/lmstudio-community/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf as the source file for the LLM.
- **Dependencies**: Required libraries for data handling and Llama queries. We used llama-cpp-python and pandas as Python Packages.

## Usage

1. **Prepare the Dataset**: Place input data in the `input` directory.
2. **Run Calibration**: Execute the calibration script to view clustering results.
3. **Generate Output**: Based on calibration results, generate refined relations and save them in the `add` and `verify` directories.

## Acknowledgments
- Our source code is based on https://swharden.com/blog/2023-07-29-ai-chat-locally-with-python/.
- We thank Wikimedia Switzerland, and particularly Ilario Valdelli, for providing computer resources for the experiment.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
