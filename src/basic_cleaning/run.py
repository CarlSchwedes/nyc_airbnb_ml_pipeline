#!/usr/bin/env python
"""
An example of a step using MLflow and Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    logger.info("Download Artifacts")
    artifact = run.use_artifact(args.input_artifact)
    artifact_local_path = artifact.file()
    
    logger.info("Load Artifacts")
    df = pd.read_csv(artifact_local_path)   
    
    logger.info("Cleaning Data...") 
    df['last_review'] = pd.to_datetime(df['last_review'])
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    
    logger.info("Create Artifact")
    file = "clean_sample.csv"
    df.to_csv(file, index=False)
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(file)
    run.log_artifact(artifact)
    ######################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="My step")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum number for price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum number for price",
        required=True
    )


    args = parser.parse_args()

    go(args)
