# Bounded penalty extension from validation evidence

The original fifteen-configuration search completed all 405 fits and exported 49,410 validation predictions. Independent scalar predictions/metrics and 405 augmented-QR/objective/gradient/Hessian certificates pass. Every family selects penalty 0.01, the smallest tested value. The maximum independent gradient is 2.58e-14 and the largest coefficient difference from the separate augmented-QR solve is 2.21e-14.

Before any 2025 inference, add penalties **0.001 and 0.0001 to each of the same three families**. This is six extra configurations and 162 additional page-fold fits, for a total of 567 trial rows. It tests whether the lower boundary conceals useful validation improvement. The two-level extension is bounded; a new boundary winner is still a finite-grid choice, not evidence of a global generalization optimum. No further range extension is planned for this experiment.

The original files and their independent check are preserved under `results/search-v1/`, with exact hashes in its search receipt and normal Git history. Incremental execution may reuse those unchanged fitted artifacts and predictions rather than refit them; a full reproducibility run refits every current candidate from scratch. Every attempted added configuration, runtime and failure is retained.

The selection objective, equal page/fold weights, candidate tie break, preprocessing, families, features, validation dates, held topics/language and 2025 test period remain unchanged. This decision uses validation boundary evidence only. No model has predicted or scored the final test period.
