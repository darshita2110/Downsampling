# Hyperspectral benchmark scenes: Indian Pines, Salinas, KSC, Botswana

Same pipeline as the Raichur runs: standardised bands, up to 8,000 random pixels (KSC has only
5,211 labelled pixels, all used), k = 12, 3 components, seed 42, Kernel PCA gamma = 1/n_bands.
Classification: 5-NN on the 3-D embedding, random 5-fold and spatial-block 5-fold (5x5 blocks).
Embeddings are fitted on all sampled pixels (labels unused): a fair method comparison, not an
out-of-sample test. One machine (single CPU core), one seed. SMACOF not run on these scenes.

Data quirks handled: KSC_data.mat stores (bands, rows, cols); Botswana has no ground truth supplied;
PaviaU has only the ground-truth file (image cube missing) so it was skipped; the `_1` files were
byte-identical duplicates.

Results: see results/benchmarks_summary.md (full table) - mean ranks (1 = best, 4 methods):
neighbourhood preservation: PCA 1.00, Isomap 2.00, Kernel PCA 3.00, LLE 4.00 (4 scenes);
spatial-block accuracy: PCA 1.33, Isomap 2.33, Kernel PCA 3.00, LLE 3.33 (3 labelled scenes);
peak memory: PCA 1.00, LLE 2.00, Kernel PCA 3.00, Isomap 4.00.

Exceptions worth stating: on KSC Kernel PCA had the best spatial-block accuracy (0.563 vs PCA 0.531),
and LLE was poor on KSC (trustworthiness 0.887, spatial accuracy 0.339).
Honest reading: PCA was at least as good as every non-linear method on neighbourhood preservation
in all four scenes; non-linear methods did not justify their cost at 3 components here.
