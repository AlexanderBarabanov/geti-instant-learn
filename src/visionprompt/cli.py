"""This module contains the VisionPrompt CLI."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
import inspect
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


from jsonargparse import ActionConfigFile, ArgumentParser, Namespace

from visionprompt.benchmark import perform_benchmark_experiment
from visionprompt.pipelines import GroundingDinoSAM
from visionprompt.pipelines.pipeline_base import Pipeline
from visionprompt.run import run_pipeline
from visionprompt.utils.args import populate_benchmark_parser


class VisionPromptCLI:
    """This class is the entry point for the VisionPrompt CLI."""

    def __init__(self) -> None:
        self.parser = ArgumentParser(description="VisionPrompt CLI", env_prefix="visionprompt")
        self._add_subcommands()
        self.execute()

    @staticmethod
    def add_run_arguments(parser: ArgumentParser) -> None:
        """Add arguments for the run subcommand."""
        # load datasets
        parser.add_subclass_arguments(Pipeline, "pipeline", default="Matcher")
        parser.add_argument(
            "--reference_image_dir", "--ref", type=str, default=None, help="Directory with reference images."
        )
        parser.add_argument(
            "--target_image_dir", "--target", type=str, required=True, help="Directory with target images."
        )
        parser.add_argument(
            "--reference_prompt_dir",
            "--ref_prompt",
            type=str,
            default=None,
            help="Directory with reference prompts (masks or points).",
        )
        parser.add_argument(
            "--points", type=str, default=None, help="Reference points as a string. e.g. [0:[640,640], -1:[200,200]]"
        )
        parser.add_argument(
            "--reference_text_prompt",
            "--text",
            type=str,
            default=None,
            help="Text prompt for grounding dino. If provided, pipeline is set to GroundingDinoSAM.",
        )
        parser.add_argument("--output_location", type=str, default=None, help="Directory to save output.")
        parser.add_argument("--chunk_size", type=int, default=5, help="Chunk size for processing target images.")

    @staticmethod
    def add_benchmark_arguments(parser: ArgumentParser) -> None:
        """Add arguments for the benchmark subcommand."""
        # TODO(Daankrol): rewrite benchmark script into a class and add arguments here  # noqa: TD003
        populate_benchmark_parser(parser)

    @staticmethod
    def add_ui_arguments(parser: ArgumentParser) -> None:
        """Add arguments for the ui subcommand."""
        parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the UI on.")
        parser.add_argument("--debug", type=bool, default=True, help="Whether to run the UI in debug mode.")
        parser.add_argument("--port", type=int, default=5050, help="Port to run the UI on.")

    @staticmethod
    def _set_text_prompt_args(cfg: Namespace) -> None:
        """Switch to GroundingDinoSAM pipeline if text prompt is provided."""
        if cfg.subcommand == "run" and cfg.run.reference_text_prompt:
            sig = inspect.signature(GroundingDinoSAM.__init__)
            gino_accepted_args = {p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD}

            cfg.run.pipeline.init_args = {
                key: value for key, value in cfg.run.pipeline.init_args.items() if key in gino_accepted_args
            }
            cfg.run.pipeline.class_path = "visionprompt.pipelines.GroundingDinoSAM"

    def execute(self) -> None:
        """Execute the CLI."""
        cfg = self.parser.parse_args()
        self._set_text_prompt_args(cfg)

        instantiated_config = self.parser.instantiate_classes(cfg)

        self._execute_subcommands(instantiated_config)

    @staticmethod
    def _visionprompt_subcommands() -> dict[str, str]:
        """Returns the subcommands and help messages for each subcommand."""
        return {
            "run": "Perform both learning and inference steps.",
            "benchmark": "Run benchmarking on the pipelines.",
            "ui": "Run the UI for the pipelines.",
        }

    def _add_subcommands(self) -> None:
        """Registers the subcommands for the CLI."""
        parser_subcommands = self.parser.add_subcommands()

        for name, description in self._visionprompt_subcommands().items():
            parser = ArgumentParser(description=description)
            self._add_common_args(parser)
            getattr(self, f"add_{name}_arguments")(parser)
            parser_subcommands.add_subcommand(name, parser)

    @staticmethod
    def _add_common_args(parser: ArgumentParser) -> None:
        """Adds common arguments for all subcommands."""
        parser.add_argument("--config", action=ActionConfigFile)

    @staticmethod
    def _execute_subcommands(config: Namespace) -> None:
        """Run the appropriate subcommand based on the config."""
        subcommand = config.subcommand
        match subcommand:
            case "run":
                if not config.run.reference_image_dir and not config.run.reference_text_prompt:
                    msg = "Either reference_image_dir or reference_text_prompt must be provided."
                    raise ValueError(msg)

                pipeline = config.run.pipeline
                run_pipeline(
                    pipeline=pipeline,
                    target_image_dir=config.run.target_image_dir,
                    reference_image_dir=config.run.reference_image_dir,
                    reference_prompt_dir=config.run.reference_prompt_dir,
                    reference_points_str=config.run.points,
                    reference_text_prompt=config.run.reference_text_prompt,
                    output_location=config.run.output_location,
                    chunk_size=config.run.chunk_size,
                )
            case "benchmark":
                perform_benchmark_experiment(config.benchmark)
            case "ui":
                from web_ui.app import app

                app.run(host=config.ui.host, debug=config.ui.debug, port=config.ui.port)
            case _:
                msg = f"Invalid subcommand: {subcommand}"
                raise ValueError(msg)


def main() -> None:
    """Main function for the CLI."""
    VisionPromptCLI()


if __name__ == "__main__":
    main()
