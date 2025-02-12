import argparse
from utils.constants import MODEL_MAP, DATASETS, ALGORITHMS


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sam_name", type=str, default="MobileSAM", choices=MODEL_MAP.keys()
    )
    parser.add_argument("--max-num-pos", type=int, default=1)
    parser.add_argument("--min-num-pos", type=int, default=1)
    parser.add_argument("--algo", type=str, default="PerSAM", choices=ALGORITHMS)
    parser.add_argument(
        "--n_shot",
        type=int,
        default=1,
        help="Number of prior images to use as references",
    )
    parser.add_argument("--dataset_name", type=str, default="PerSeg", choices=DATASETS)
    parser.add_argument("--save", action="store_true", help="Save results to disk")
    parser.add_argument(
        "--show", action="store_true", help="Show results during processing"
    )
    parser.add_argument(
        "--post_refinement", action="store_true", help="Apply post refinement"
    )
    parser.add_argument(
        "--mask_gen",
        dest="mask_generation_method",
        type=str,
        default="point-by-point",
        choices=["point-by-point", "one-go"],
        help="Mask generation method",
    )
    parser.add_argument(
        "--selection_on_similarity_maps",
        type=str,
        default="per-map",
        choices=["per-map", "stacked-maps"],
        help="Apply point selection on each similarity map or stack and reduce maps to one first.",
    )
    parser.add_argument(
        "--target_guided_attention",
        action="store_true",
        help="Use target guided attention for the SAM model. This passes the target similarity matrix and reference features to the decoder",
    )
    parser.add_argument(
        "--class_name", type=str, default=None, help="Filter on class name"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output data"
    )
    parser.add_argument(
        "--n_clusters",
        "--num_clusters",
        dest="num_clusters",
        type=int,
        default=1,
        help="Number of clusters for PartAwareSAM, if 1 use mean of all features",
    )

    # Matcher specific arguments
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha for Matcher")
    parser.add_argument("--beta", type=float, default=0.0, help="Beta for Matcher")
    parser.add_argument("--exp", type=float, default=0.0, help="Exponent for Matcher")
    parser.add_argument("--use_box", action="store_true", help="Use box for Matcher")
    parser.add_argument(
        "--sample_range",
        type=lambda x: tuple(map(int, x.split(","))),
        default=(4, 6),
        help="Sample range for Matcher (min,max). Example: --sample_range 4,6",
    )
    parser.add_argument(
        "--max_sample_iterations",
        type=int,
        default=30,
        help="Max sample iterations for Matcher",
    )
    parser.add_argument("--emd_filter", type=float, default=0.0, help="Use emd_filter")
    parser.add_argument(
        "--purity_filter", type=float, default=0.0, help="Use purity_filter"
    )
    parser.add_argument(
        "--coverage_filter", type=float, default=0.0, help="Use coverage_filter"
    )
    parser.add_argument(
        "--use_score_filter",
        type=bool,
        default=True,
        help="Use score_filter",
    )
    parser.add_argument(
        "--num_merging_masks",
        type=int,
        default=10,
        help="topK masks to merge",
    )
    parser.add_argument(
        "--topk_scores_threshold",
        type=float,
        default=0.7,
        help="topK scores threshold",
    )
    parser.add_argument(
        "--deep_score_filter",
        type=float,
        default=0.33,
        help="deep score filter",
    )
    parser.add_argument(
        "--deep_score_norm_filter",
        type=float,
        default=0.1,
        help="deep score norm filter",
    )

    args = parser.parse_args()
    return args
