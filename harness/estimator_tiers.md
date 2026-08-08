# Baseline feasibility tiers (mechanically derived)

Population: all 50 rows of upstream `examples/configs/estimators/default_estimators.yaml`. Each name+cfg pair is a distinct row. Tier is assigned by predicates over each estimator's *resolved transitive dependency set*, not by a list of names.

| Tier | Count | Meaning |
|---|---:|---|
| `needs_train_data` | 5 | Fits statistics on a train/background split. Excluded: the assignment forbids supervised training, and this is fitting on held-out train/background data in all but name. |
| `needs_external_corpus` | 1 | Downloads a large external corpus or artifact at init. Excluded: not feasible on a free T4 session and not required by any constraint. |
| `needs_sampling` | 29 | Requires multiple sampled generations per input. Excluded from the primary table: cost is a multiple of the single-pass budget, so it is not a matched-compute comparison. Reported separately at n=300. |
| `single_pass_plus_aux_model` | 1 | One generation plus a second neural model (NLI cross-encoder) over that generation -- no sampling. INCLUDED in the primary table as its own row, flagged: it is materially cheaper than the sampling tier but is not a pure single-pass method. |
| `requires_bs1_under_fp16` | 4 | Consumes attention maps, which forces output_attentions=True. That disables transformers' left-padding NaN guard (AttentionMaskConverter._unmask_unattended runs only when _attn_implementation == 'sdpa' AND output_attentions is False), so under fp16 a fully-masked attention row overflows to -inf and softmax returns NaN -- observed as 108/150 generations collapsing to token 0. NOT excluded: run separately at batch_size=1 with eager attention, where no padding exists and therefore no attention row is ever fully masked. Reported as a secondary table at n=300. |
| `single_pass_cheap` | 10 | Primary baseline set. |

## needs_train_data  (5)

| Estimator | cfg | Resolved calculators | Flags |
|---|---|---|---|
| `MahalanobisDistanceSeq_decoder` | - | GreedyProbsCalculator,TrainingStatisticExtractionCalculator | needs_external_corpus,needs_train_data |
| `PPLMDSeq_decoder` | md_type=MD | GreedyProbsCalculator,TrainingStatisticExtractionCalculator | needs_external_corpus,needs_train_data |
| `PPLRMDSeq_decoder` | md_type=RMD | GreedyProbsCalculator,TrainingStatisticExtractionCalculator | needs_external_corpus,needs_train_data |
| `RDESeq_decoder` | - | GreedyProbsCalculator,TrainingStatisticExtractionCalculator | needs_external_corpus,needs_train_data |
| `RelativeMahalanobisDistanceSeq_decoder` | - | GreedyProbsCalculator,TrainingStatisticExtractionCalculator | needs_external_corpus,needs_train_data |

## needs_external_corpus  (1)

| Estimator | cfg | Resolved calculators | Flags |
|---|---|---|---|
| `Focus` | gamma=0.9;idf_dataset=krisbailey/RedPajama-Data-V2-100M;idf_dataset_size=-1;idf_dataset_text_column=raw_content;idf_seed=42;model_name=${model.path};p=0.01;path=${cache_path}/focus/${model.path}/token_idf.pkl;spacy_path=en_core_web_sm;trust_remote_code=False | - | needs_external_corpus |

## needs_sampling  (29)

| Estimator | cfg | Resolved calculators | Flags |
|---|---|---|---|
| `CocoaMSP` | - | GreedyCrossEncoderSimilarityMatrixCalculator,GreedyProbsCalculator,InitialStateCalculator,SamplingGenerationCalculator | needs_auxiliary_model,needs_sampling |
| `CocoaMTE` | - | EntropyCalculator,GreedyCrossEncoderSimilarityMatrixCalculator,GreedyProbsCalculator,InitialStateCalculator,SamplingGenerationCalculator | needs_auxiliary_model,needs_sampling |
| `CocoaPPL` | - | GreedyCrossEncoderSimilarityMatrixCalculator,GreedyProbsCalculator,InitialStateCalculator,SamplingGenerationCalculator | needs_auxiliary_model,needs_sampling |
| `DegMat_Jaccard_score` | similarity_score=Jaccard_score | SamplingGenerationCalculator | needs_sampling |
| `DegMat_NLI_score_contra` | affinity=contra;similarity_score=NLI_score | SamplingGenerationCalculator,SemanticMatrixCalculator | needs_auxiliary_model,needs_sampling |
| `DegMat_NLI_score_entail` | affinity=entail;similarity_score=NLI_score | SamplingGenerationCalculator,SemanticMatrixCalculator | needs_auxiliary_model,needs_sampling |
| `Eccentricity_Jaccard_score` | similarity_score=Jaccard_score | SamplingGenerationCalculator | needs_sampling |
| `Eccentricity_NLI_score_contra` | affinity=contra;similarity_score=NLI_score | SamplingGenerationCalculator,SemanticMatrixCalculator | needs_auxiliary_model,needs_sampling |
| `Eccentricity_NLI_score_entail` | affinity=entail;similarity_score=NLI_score | SamplingGenerationCalculator,SemanticMatrixCalculator | needs_auxiliary_model,needs_sampling |
| `EigValLaplacian_Jaccard_score` | similarity_score=Jaccard_score | SamplingGenerationCalculator | needs_sampling |
| `EigValLaplacian_NLI_score_contra` | affinity=contra;similarity_score=NLI_score | SamplingGenerationCalculator,SemanticMatrixCalculator | needs_auxiliary_model,needs_sampling |
| `EigValLaplacian_NLI_score_entail` | affinity=entail;similarity_score=NLI_score | SamplingGenerationCalculator,SemanticMatrixCalculator | needs_auxiliary_model,needs_sampling |
| `EigenScore` | - | SamplingGenerationCalculator | needs_sampling |
| `KernelLanguageEntropy` | - | SamplingGenerationCalculator,SemanticMatrixCalculator | needs_auxiliary_model,needs_sampling |
| `LUQ` | - | SamplingGenerationCalculator,SemanticMatrixCalculator | needs_auxiliary_model,needs_sampling |
| `LexicalSimilarity_BLEU` | metric=BLEU | SamplingGenerationCalculator | needs_sampling |
| `LexicalSimilarity_rouge1` | metric=rouge1 | SamplingGenerationCalculator | needs_sampling |
| `LexicalSimilarity_rouge2` | metric=rouge2 | SamplingGenerationCalculator | needs_sampling |
| `LexicalSimilarity_rougeL` | metric=rougeL | SamplingGenerationCalculator | needs_sampling |
| `MonteCarloNormalizedSequenceEntropy` | - | SamplingGenerationCalculator | needs_sampling |
| `MonteCarloSequenceEntropy` | - | SamplingGenerationCalculator | needs_sampling |
| `NumSemSets` | - | SamplingGenerationCalculator,SemanticMatrixCalculator | needs_auxiliary_model,needs_sampling |
| `PTrueSampling` | - | GreedyProbsCalculator,SamplingGenerationCalculator,SamplingPromptCalculator | needs_sampling |
| `SAR` | - | CrossEncoderSimilarityMatrixCalculator,GreedyProbsCalculator,InitialStateCalculator,SamplingGenerationCalculator | needs_auxiliary_model,needs_sampling |
| `SemanticDensity` | - | ConcatGreedySemanticMatrixCalculator,GreedyProbsCalculator,InitialStateCalculator,RawInputCalculator,SamplingGenerationCalculator | needs_auxiliary_model,needs_sampling |
| `SemanticEntropy` | - | SamplingGenerationCalculator,SemanticClassesCalculator,SemanticMatrixCalculator | needs_auxiliary_model,needs_sampling |
| `SemanticEntropy (Direct)` | entropy_estimation=direct | SamplingGenerationCalculator,SemanticClassesCalculator,SemanticMatrixCalculator | needs_auxiliary_model,needs_sampling |
| `SentenceSAR` | - | CrossEncoderSimilarityMatrixCalculator,GreedyProbsCalculator,InitialStateCalculator,SamplingGenerationCalculator | needs_auxiliary_model,needs_sampling |
| `TokenSAR` | - | CrossEncoderSimilarityMatrixCalculator,GreedyProbsCalculator,InitialStateCalculator,SamplingGenerationCalculator | needs_auxiliary_model,needs_sampling |

## single_pass_plus_aux_model  (1)

| Estimator | cfg | Resolved calculators | Flags |
|---|---|---|---|
| `CCP` | - | GreedyAlternativesNLICalculator,GreedyProbsCalculator | needs_auxiliary_model,single_pass_plus_aux_model |

## requires_bs1_under_fp16  (4)

| Estimator | cfg | Resolved calculators | Flags |
|---|---|---|---|
| `AttentionScore (layer=None)` | gen_only=False | AttentionForwardPassCalculator,GreedyProbsCalculator | needs_attention |
| `CSL` | - | AttentionElicitingPromptCalculator,GreedyProbsCalculator | needs_attention |
| `RAUQ` | alpha=0.2;use_entropy=False | GreedyProbsCalculator | needs_attention |
| `RAUQ (entropy)` | alpha=0.8;use_entropy=True | EntropyCalculator,GreedyProbsCalculator | needs_attention |

## single_pass_cheap  (10)

| Estimator | cfg | Resolved calculators | Flags |
|---|---|---|---|
| `BoostedProbSequence` | - | GreedyProbsCalculator | - |
| `FisherRao` | - | GreedyProbsCalculator | - |
| `MaximumSequenceProbability` | - | GreedyProbsCalculator | - |
| `MeanConditionalPointwiseMutualInformation` | - | EntropyCalculator,GreedyLMProbsCalculator,GreedyProbsCalculator | - |
| `MeanPointwiseMutualInformation` | - | GreedyLMProbsCalculator,GreedyProbsCalculator | - |
| `MeanTokenEntropy` | - | EntropyCalculator,GreedyProbsCalculator | - |
| `PTrue` | - | GreedyProbsCalculator,PromptCalculator | - |
| `Perplexity` | - | GreedyProbsCalculator | - |
| `RenyiNeg` | - | GreedyProbsCalculator | - |
| `SelfCertainty` | - | GreedyProbsCalculator | - |
