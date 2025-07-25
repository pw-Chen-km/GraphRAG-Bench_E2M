## Overview

This is the GraphRAG-bench project. The system uses a merged configuration approach to support multiple GraphRAG methods with unified configuration management. We recommend that you directly use the official gitHub repositories of each RAG method.

## Usage

### 1. Basic Usage

```bash
# Use the refactored main program with merged configuration
python main.py -opt Option/merged_config.yaml -dataset_name your_dataset
```

### 2. Advanced Options

```bash
# Force rebuild
python main.py -opt Option/merged_config.yaml -dataset_name your_dataset --force_rebuild

# Skip evaluation
python main.py -opt Option/merged_config.yaml -dataset_name your_dataset --skip_evaluation

# Show system information
python main.py -opt Option/merged_config.yaml -dataset_name your_dataset --show_info
```

### 3. Acknowledgments
We sincerely thank the authors of DIGMON, RAPTOR, HippoRAG, GFM-RAG, LightRAG, G-Retriever, GraphRAG, KGP, DALK and ToG.
